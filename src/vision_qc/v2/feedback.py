"""Build isolated candidates from audited live labels, never overwrite an active run."""
from pathlib import Path
import json

from ..data import Sample, load_manifest, write_manifest
from ..pipeline import dump_json
from ..store import ReviewStore
from .evidence import pixel_hash
from .pipeline import InspectorV2, train_v2
from .secondary import Segmenter
from .learning import attach_secondary


def retrain_feedback(run, state, output, steps=120):
    run,state,output = Path(run),Path(state),Path(output)
    if output.exists():
        raise FileExistsError(output)
    inspector = InspectorV2(run)
    original = load_manifest(inspector.meta["manifest"])
    used = {pixel_hash(s.path) for s in original}
    provenance = json.loads((run/"provenance.json").read_text(encoding="utf-8"))
    used.update(r["pixels"] for r in provenance)
    previous_feedback = {}
    if (run/"feedback-snapshot.json").exists():
        previous_feedback = {r["id"]:r for r in json.loads((run/"feedback-snapshot.json").read_text(encoding="utf-8"))}
        used.update(r["pixels"] for r in previous_feedback.values())
    store = ReviewStore(state/"reviews.sqlite")
    masks = store.masks()
    feedback,ledger = [],[]
    for row in store.list("labeled",100000):
        if row["result"]["category"] != inspector.meta["category"]:
            continue
        path = Path(row["image_path"])
        digest = pixel_hash(path)
        previous = previous_feedback.get(row["id"])
        if previous and (previous["label"] != row["label"] or previous["revision"] != row["revision"]):
            raise ValueError("Previously trained feedback was corrected; rebuild from an audited corrected manifest")
        if digest in used:
            continue
        mask = masks.get(row["id"])
        if row["label"]==1 and mask is None:
            continue
        used.add(digest)
        feedback.append(Sample("feedback-"+row["id"],path,inspector.meta["category"],
                               "train" if row["label"]==0 else "pool",row["label"],"feedback-"+row["id"],
                               Path(mask["path"]) if mask else None))
        ledger.append(dict(id=row["id"],label=row["label"],revision=row["revision"],pixels=digest,
                           mask=mask,history=store.history(row["id"])))
    if not feedback:
        raise ValueError("No new eligible labels; defects require current pixel masks, and holdout images are excluded")
    # Prepare beside the candidate so failed training leaves a reviewable snapshot.
    work = output.with_name(output.name+"-training")
    work.mkdir(parents=True,exist_ok=False)
    manifest = work/"manifest.csv"
    # Reusing old holdout labels is research only; fresh acceptance is required later.
    write_manifest(manifest,[dict(id=s.id,path=str(s.path),category=s.category,split=s.split,label=s.label,
        group=s.group,mask=str(s.mask) if s.mask else "") for s in original+feedback])
    config = dict(inspector.meta["config"])
    train_v2(manifest,work/"primary",config,independent=False)
    supervised = [s for s in original if s.split=="train"][:16]+feedback
    secondary = Segmenter().fit(supervised,steps=steps)
    meta = attach_secondary(work/"primary",output,secondary,[s.id for s in supervised])
    dump_json(output/"feedback-snapshot.json",list(previous_feedback.values())+ledger)
    return dict(run=str(output),accepted_labels=len(feedback),normal_labels=sum(s.label==0 for s in feedback),
                defect_masks=sum(s.label==1 for s in feedback),version=meta["version"],release_approved=False)
