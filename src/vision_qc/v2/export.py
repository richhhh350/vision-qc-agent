"""Fixed-tile ONNX export with native-map and decision parity verification."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..data import load_manifest
from ..model import Detection
from ..pipeline import dump_json
from .geometry import Geometry
from .pipeline import InspectorV2
from .evidence import policy_hash


class OnnxDetector:
    backend = "onnx"

    def __init__(self, run):
        import onnxruntime as ort
        run = Path(run)
        self.contract = json.loads((run/"onnx-contract.json").read_text(encoding="utf-8"))
        if hashlib.sha256((run/"detector.onnx").read_bytes()).hexdigest() != self.contract["sha256"]:
            raise ValueError("ONNX checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.contract["threads"]
        self.session = ort.InferenceSession(str(run/"detector.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
        self.geometry = Geometry(**self.contract["geometry"])

    def predict(self, image):
        start = time.perf_counter()
        pixels,records = self.geometry.prepare(image)
        scores,maps,embeddings = [],[],[]
        for tile in pixels:
            score,heatmap,embedding = self.session.run(None,{"image":tile[None]})
            scores.append(float(score.max())); maps.append(heatmap); embeddings.append(embedding)
        self.last_timing = dict(onnx_ms=(time.perf_counter()-start)*1000,tiles=len(records))
        return Detection(max(scores),self.geometry.restore(np.concatenate(maps),records,image.size),
                         np.concatenate(embeddings).mean(0))


def export_v2(run, manifest, limit=32, atol=1e-3, rtol=1e-3, verify_only=False):
    import torch
    run = Path(run)
    inspector = InspectorV2(run)
    detector = inspector.model
    if inspector.meta["backend"] not in {"patchcore","efficientad"}:
        raise ValueError("Use the v1 exporter for the legacy backend")
    if (run/"detector.onnx").exists() and not verify_only:
        raise FileExistsError("ONNX artifact already exists; use a separate model package")
    if limit < 1:
        raise ValueError("Positive parity sample count required")
    samples = [s for s in load_manifest(manifest) if s.split=="dev"]
    if not samples:
        raise ValueError("Separate dev samples required for ONNX parity")

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = detector.network
            self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])
        def forward(self,x):
            output = self.network((x-self.mean)/self.std if detector.backend=="patchcore" else x)
            return output.pred_score,output.anomaly_map,detector._embedding

    size = detector.geometry.size
    if not verify_only:
        torch.onnx.export(Wrapper().eval(),torch.zeros(1,3,size,size),str(run/"detector.onnx"),
            input_names=["image"],output_names=["score","heatmap","embedding"],
            opset_version=17,dynamo=False)
        contract = dict(schema_version=2,source_policy_sha256=policy_hash(inspector.meta),
            sha256=hashlib.sha256((run/"detector.onnx").read_bytes()).hexdigest(),
            geometry=detector.geometry.to_dict(),threads=detector.config["threads"],
            input="RGB float32 [0,1]; fixed one-tile BCHW; normalization embedded",
            source_backend=detector.backend,score_aggregation="max per tile",map_aggregation="mean overlap",
            release_approved=False,note="Changing runtime requires recalibration and fresh whole-policy acceptance")
        dump_json(run/"onnx-contract.json",contract)
    else:
        contract=json.loads((run/"onnx-contract.json").read_text(encoding="utf-8"))
        if contract["source_policy_sha256"]!=policy_hash(inspector.meta):
            raise ValueError("ONNX artifact belongs to another policy")
    onnx = OnnxDetector(run)
    # Select threshold-near samples using dev scores only, plus evenly spaced IDs.
    scored = []
    thresholds = []
    for kind in ("normal","defect"):
        values = sorted(inspector.meta["conformal"][kind+"_scores"])
        required = int(np.floor(inspector.meta["conformal"]["alpha_"+kind]*(len(values)+1)))
        if required:
            thresholds.append(values[-required] if kind=="normal" else values[required-1])
    thresholds = np.asarray(thresholds)
    for sample in samples:
        with Image.open(sample.path) as image:
            detected = detector.predict(image)
        margin = float(np.min(np.abs(thresholds-detected.score))) if len(thresholds) else 0
        scored.append((margin,sample))
    near = [s for _,s in sorted(scored,key=lambda pair:pair[0])[:max(1,limit//2)]]
    ids = {s.id for s in near}
    for i in np.linspace(0,len(samples)-1,min(limit,len(samples)),dtype=int):
        if len(near)>=limit:
            break
        if samples[i].id not in ids:
            near.append(samples[i]); ids.add(samples[i].id)
    rows = []
    for sample in near:
        with Image.open(sample.path) as image:
            reference = detector.predict(image)
            other = onnx.predict(image)
        raw_a = inspector.router.predict(reference.score)["route"]
        raw_b = inspector.router.predict(other.score)["route"]
        rows.append(dict(id=sample.id,torch_score=reference.score,onnx_score=other.score,
            score_abs_error=abs(reference.score-other.score),
            map_max_abs_error=float(np.max(np.abs(reference.heatmap-other.heatmap))),
            map_allclose=bool(np.allclose(reference.heatmap,other.heatmap,atol=atol,rtol=rtol)),
            score_allclose=bool(np.isclose(reference.score,other.score,atol=atol,rtol=rtol)),
            embedding_allclose=bool(np.allclose(reference.embedding,other.embedding,atol=atol,rtol=rtol)),
            raw_route_equal=raw_a==raw_b,torch_raw_route=raw_a,onnx_raw_route=raw_b))
    report = dict(n=len(rows),atol=atol,rtol=rtol,source_policy_sha256=policy_hash(inspector.meta),
        passed=all(r["map_allclose"] and r["score_allclose"] and r["embedding_allclose"] and r["raw_route_equal"] for r in rows),
        raw_route_flips=sum(not r["raw_route_equal"] for r in rows),samples=rows,
        selection="dev samples nearest the two actual conformal decision boundaries, plus evenly spaced samples",
        decision_boundaries=thresholds.tolist(),model_bytes=(run/"detector.onnx").stat().st_size,
        note="Parity does not transfer a release certificate; ONNX remains an unapproved candidate")
    dump_json(run/"onnx-parity.json",report)
    return report
