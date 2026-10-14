import copy
import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from vision_qc.data import synthetic_image, write_manifest, load_manifest
from vision_qc.api import create_app
from vision_qc.metrics import auroc, binomial_upper
from vision_qc.v2.evidence import certificate, check_certificate, fingerprints, policy_hash, check_weights
from vision_qc.v2.geometry import Geometry
from vision_qc.v2.metrics import NativePixelMetrics
from vision_qc.v2.pipeline import train_v2, InspectorV2, evaluate_v2
from vision_qc.v2.registry import Registry
from vision_qc.v2.learning import acquire


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2")
    rng = np.random.default_rng(999)
    rows = []
    for split,n in (("train",8),("prob_fit",6),("calibration",8),("acceptance",8),("dev",6),("test",6),("pool",8)):
        for i in range(n):
            label = 0 if split=="train" else i%2
            image,mask = synthetic_image(rng,bool(label))
            name = f"{split}-{i}"
            image.save(root/f"{name}.png"); mask.save(root/f"{name}-mask.png")
            rows.append(dict(id=name,path=str(root/f"{name}.png"),mask=str(root/f"{name}-mask.png"),
                             split=split,label=label,category="test_product",group=name))
    manifest = root/"manifest.csv"
    write_manifest(manifest,rows)
    train_v2(manifest,root/"model",dict(detector=dict(backend="patch_memory_cpu",bank_size=32),synthetic=True))
    return root


@pytest.mark.parametrize("mode",["letterbox","tiles"])
@pytest.mark.parametrize("color",["RGB","L","RGBA"])
def test_native_geometry(mode,color):
    image = Image.new(color,(73,191))
    geometry = Geometry(size=64,mode=mode,max_tiles=100)
    tensor,records = geometry.prepare(image)
    restored = geometry.restore(np.ones((len(tensor),1,16,16),np.float32),records,image.size)
    assert restored.shape == (191,73)
    assert np.allclose(restored,1)
    # Edge tile stays at the right/bottom original-image coordinates.
    if mode=="tiles":
        values = np.zeros((len(tensor),1,64,64),np.float32); values[-1]=1
        heatmap = geometry.restore(values,records,image.size)
        assert heatmap[-1,-1]==1 and heatmap[0,0]==0


def test_geometry_invalid_and_limit():
    with pytest.raises(ValueError):
        Geometry(size=33)
    with pytest.raises(ValueError,match="tiles"):
        Geometry(size=32,mode="tiles",max_tiles=1).prepare(Image.new("RGB",(100,100)))


def valid_certificate_meta():
    meta = dict(schema_version=2,backend="patchcore",category="widget",synthetic=False,
                conformal=dict(alpha_defect=.01),evidence=dict(calibration_sha256="a"*64,acceptance_sha256="b"*64))
    meta["certificate"] = certificate(meta,dict(n_defect=300,missed_defects=0,fnr_upper_95=binomial_upper(0,300),false_alarm_rate=.02),
                                       "a"*64,"b"*64,independent=True)
    return meta


def test_certificate_fail_closed():
    meta = valid_certificate_meta()
    assert check_certificate(meta)==(True,"approved")
    for mutation in (lambda m:m.update(category="other"),lambda m:m["conformal"].update(alpha_defect=.02),
                     lambda m:m.update(audit_fraction=.2),lambda m:m["evidence"].update(acceptance_sha256="c"*64),
                     lambda m:m["certificate"].update(n_defect=50),lambda m:m["certificate"].update(fnr_upper_95=float("nan")),
                     lambda m:m["certificate"].update(expires_at="broken"),lambda m:m["certificate"].pop("category"),
                     lambda m:m["certificate"].update(false_alarm_rate=.3),lambda m:m.update(synthetic=True)):
        changed=copy.deepcopy(meta); mutation(changed)
        assert not check_certificate(changed)[0]
    assert not check_certificate(meta,datetime.now(timezone.utc)+timedelta(days=31))[0]


def test_decoded_duplicate_is_not_independent(v2_run,tmp_path):
    from dataclasses import replace
    first=load_manifest(v2_run/"manifest.csv")[0]
    copy_path=tmp_path/"copy.bmp"
    with Image.open(first.path) as image:
        image.save(copy_path)
    with pytest.raises(ValueError,match="Decoded"):
        fingerprints([first,replace(first,id="other",path=copy_path,split="test",group="other")])


def test_native_metric_agrees_with_exact_ranks():
    rng=np.random.default_rng(4)
    heat=rng.random((35,49)); mask=heat>.6
    metric=NativePixelMetrics()
    metric.update(mask,heat,.7)
    result=metric.result()
    assert abs(result["pixel_auroc_binned"]-auroc(mask.ravel(),heat.ravel()))<=result["pixel_auroc_max_binning_error"]+1e-12
    assert 0<=result["aupro_0_30_binned"]<=1


def test_v2_service_masks_and_gate(v2_run,tmp_path):
    client=TestClient(create_app(v2_run/"model",tmp_path))
    assert client.get("/model").json()["schema_version"]==2
    raw=(v2_run/"test-1.png").read_bytes()
    row=client.post("/inspect",files={"file":("x.png",raw)},data={"category":"test_product"}).json()
    assert row["result"]["route"]!="pass"
    assert row["result"]["heatmap_coordinates"]=="original_image"
    url=f"/reviews/{row['id']}"
    assert client.post(url+"/feedback",json=dict(label=1,reviewer="test",revision=0)).status_code==200
    mask=(v2_run/"test-1-mask.png").read_bytes()
    assert client.post(url+"/mask",files={"file":("m.png",mask)},data={"revision":1}).status_code==200
    assert client.post(url+"/mask",files={"file":("m.png",mask)},data={"revision":0}).status_code==422
    from vision_qc.store import ReviewStore
    store=ReviewStore(tmp_path/"reviews.sqlite")
    assert row["id"] in store.masks()
    assert client.post(url+"/feedback",json=dict(label=0,reviewer="fix",revision=1)).status_code==200
    assert row["id"] not in store.masks()
    duplicate=client.post("/inspect",files={"file":("x.png",raw)},data={"category":"unknown"})
    assert duplicate.json()["result"]["route"]=="review"
    assert client.get("/training-jobs/bad").status_code==404
    report=evaluate_v2(v2_run/"model",v2_run/"manifest.csv")
    assert report["n"]==6 and report["pixel_evaluation_images"]==6


def test_registry_promotion_rollback_and_tampering(v2_run,tmp_path):
    import shutil
    registry=Registry(tmp_path/"registry")
    first=registry.register(v2_run/"model")
    registry.activate(first["version"])
    with pytest.raises(ValueError,match="release"):
        registry.activate(first["version"],require_release=True)
    other=tmp_path/"other"; shutil.copytree(v2_run/"model",other)
    meta=json.loads((other/"model.json").read_text(encoding="utf-8")); meta["version"]="second"
    (other/"model.json").write_text(json.dumps(meta),encoding="utf-8")
    registry.register(other); registry.activate("second")
    assert registry.rollback()["version"]==first["version"]
    (other/"detector.npz").write_bytes(b"broken")
    with pytest.raises(ValueError,match="checksum"):
        registry.activate("second")


def test_acquisition_has_no_oracle_and_audit_probabilities():
    items=[dict(id=str(i),embedding=[i,i*i],score=i/10,uncertainty=1-abs(i-5)/5) for i in range(10)]
    for strategy in ("random","uncertainty","coreset","hybrid"):
        chosen=acquire(items,5,strategy,42)
        assert chosen==acquire(items,5,strategy,42)
        assert len({r["id"] for r in chosen})==5
        assert all(0<r["conditional_draw_probability"]<=1 for r in chosen)
    with pytest.raises(ValueError,match="oracle"):
        acquire([dict(items[0],label=1)],1,"random",42)


def test_secondary_uses_defect_masks(v2_run,tmp_path):
    pytest.importorskip("torch")
    from vision_qc.v2.secondary import Segmenter
    from vision_qc.v2.learning import attach_secondary
    from dataclasses import replace
    selected=[s for s in load_manifest(v2_run/"manifest.csv") if s.split=="pool"][:4]
    with pytest.raises(ValueError,match="masks"):
        Segmenter().fit([replace(selected[1],mask=None)],steps=1)
    segmenter=Segmenter().fit(selected,steps=2)
    assert segmenter.training_record["defect_masks"]==2
    assert segmenter.training_record["positive_tiles_seen"]>=1
    attach_secondary(v2_run/"model",tmp_path/"candidate",segmenter,[s.id for s in selected])
    inspector=InspectorV2(tmp_path/"candidate")
    with Image.open(selected[1].path) as image:
        result,_=inspector.inspect(image,"unknown")
    assert result["secondary"]["role"]=="human_advice_only"
    assert not inspector.meta["release_gate"]["approved"]
