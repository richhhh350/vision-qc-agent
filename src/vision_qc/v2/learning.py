"""Reproducible pool simulation: acquire first, reveal labels second.

Development curves are exploratory and never production risk certificates.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..data import load_manifest
from ..metrics import auroc
from ..pipeline import dump_json
from .evidence import check_weights, fingerprints, policy_hash
from .pipeline import InspectorV2


def acquire(items, budget, strategy, seed, audit_fraction=.1):
    """Sequential epsilon mixture, with exact per-draw conditional probabilities.

    Accepts predictions/features only. A label/mask on an acquisition item is an error.
    Diversity measures distance to selected points within this batch.
    """
    if strategy not in {"random","uncertainty","coreset","hybrid"}:
        raise ValueError("Unknown strategy")
    if any("label" in item or "mask" in item for item in items):
        raise ValueError("Acquisition must not receive oracle labels or masks")
    if budget < 0 or not 0 <= audit_fraction <= 1:
        raise ValueError("Invalid acquisition budget/audit fraction")
    if not items:
        return []
    rng = np.random.default_rng(seed)
    features = np.asarray([x["embedding"] for x in items],dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite embeddings")
    features = (features-features.mean(0))/np.maximum(features.std(0),1e-6)
    uncertainty = np.asarray([x["uncertainty"] for x in items])
    remaining, chosen, ledger = list(range(len(items))), [], []
    for _ in range(min(budget,len(items))):
        candidates = remaining
        if strategy == "hybrid":
            candidates = sorted(remaining,key=lambda i:(-uncertainty[i],items[i]["id"]))[:max(1,len(remaining)//2)]
        if strategy == "uncertainty":
            preferred = max(remaining,key=lambda i:uncertainty[i])
        elif strategy in {"coreset","hybrid"}:
            if chosen:
                distance = ((features[candidates,None]-features[chosen][None])**2).sum(-1).min(1)
            else:
                distance = (features[candidates]**2).sum(1)
            preferred = candidates[int(np.argmax(distance))]
        else:
            preferred = remaining[0]
        epsilon = 1.0 if strategy == "random" else audit_fraction
        branch = "random_audit" if rng.random() < epsilon else strategy
        index = int(rng.choice(remaining)) if branch == "random_audit" else preferred
        probability = epsilon/len(remaining)+(1-epsilon)*int(index==preferred)
        ledger.append(dict(id=items[index]["id"],source=branch,
            conditional_draw_probability=probability,remaining=len(remaining),
            score=items[index]["score"],uncertainty=float(uncertainty[index])))
        remaining.remove(index); chosen.append(index)
    return ledger


def attach_secondary(run, output, segmenter, training_ids):
    import shutil
    run, output = Path(run),Path(output)
    meta = json.loads((run/"model.json").read_text(encoding="utf-8"))
    check_weights(run,meta)
    output.mkdir(parents=True,exist_ok=False)
    for name in meta["weights"]:
        if name != "secondary.pt":
            shutil.copy2(run/name,output/name)
    for name in ("config.json","provenance.json"):
        if (run/name).exists():
            shutil.copy2(run/name,output/name)
    segmenter.save(output/"secondary.pt")
    meta["weights"]["secondary.pt"] = hashlib.sha256((output/"secondary.pt").read_bytes()).hexdigest()
    meta["secondary"] = dict(role="human_advice_only",config=segmenter.config,training=segmenter.training_record,
                           training_ids=training_ids)
    meta["parent_version"] = meta["version"]
    meta["version"] = "v2-"+policy_hash(meta)[:16]
    meta["certificate"] = None
    meta["release_gate"].update(approved=False,reason="fresh_whole_policy_acceptance_required")
    dump_json(output/"model.json",meta)
    return meta


def run_learning(run, manifest, output, budgets=(10,20,40), seeds=(17,42,2026), steps=200, target=.96):
    from .secondary import Segmenter
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    samples = load_manifest(manifest)
    pool = {s.id:s for s in samples if s.split=="pool"}
    dev = [s for s in samples if s.split=="dev"]
    if not dev or any(s.label is None for s in pool.values()):
        raise ValueError("Simulation requires labeled oracle pool and separate dev split")
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets or budgets[0]<1 or budgets[-1]>len(pool):
        raise ValueError("Budgets must fit pool; no silent budget truncation")
    records, fingerprint = fingerprints(samples)
    inspector = InspectorV2(run)
    if {s.category for s in samples} != {inspector.meta["category"]}:
        raise ValueError("Learning category mismatch")
    output.mkdir(parents=True)
    initial = [s for s in samples if s.split=="train"][:16]
    # Predictions prepared without exposing pool labels or mask paths to acquisition.
    base = []
    for sample in pool.values():
        with Image.open(sample.path) as image:
            result,_ = inspector.inspect(image,sample.category)
        base.append(dict(id=sample.id,embedding=result["embedding"],score=result["score"],
                         uncertainty=min(result["p_normal"],result["p_defect"])))
    dev_base = []
    for sample in dev:
        with Image.open(sample.path) as image:
            dev_base.append(inspector.inspect(image,sample.category)[0])
    dump_json(output/"acquisition-features.json",base)
    fixed_labels = sum(s.split in {"train","prob_fit","calibration","acceptance","validation"} for s in samples)
    curve = []
    for seed in seeds:
        for strategy in ("random","uncertainty","coreset","hybrid"):
            selected, ledger, model = [],[],None
            for budget in budgets:
                acquisition_version = inspector.version if model is None else f"{strategy}-{seed}-{len(selected)}"
                candidates = [dict(item) for item in base if item["id"] not in selected]
                if model:
                    for item in candidates:
                        with Image.open(pool[item["id"]].path) as image:
                            item["score"] = model.predict(image)["score"]
                        item["uncertainty"] = 1-abs(2*item["score"]-1)
                acquired = acquire(candidates,budget-len(selected),strategy,seed+budget)
                # Save the acquisition decision BEFORE looking up any oracle label.
                stem = f"{strategy}-{seed}-{budget}"
                dump_json(output/f"{stem}-selection.json",acquired)
                for entry in acquired:
                    sample = pool[entry["id"]]
                    selected.append(sample.id)
                    ledger.append(dict(**entry,label=sample.label,mask_revealed=bool(sample.label==1 and sample.mask),
                        label_source="public_training_pool_oracle",annotation_seconds=None,
                        model_version=acquisition_version))
                model = Segmenter(seed=seed).fit(initial+[pool[i] for i in selected],steps=steps)
                model.save(output/f"{stem}.pt")
                scores = []
                for sample in dev:
                    with Image.open(sample.path) as image:
                        scores.append(model.predict(image)["score"])
                labels = [s.label for s in dev]
                # Cascade advice evaluated on review subset separately; all live routes stay fixed.
                review = [i for i,r in enumerate(dev_base) if r["route"]=="review"]
                row = dict(seed=seed,strategy=strategy,budget=budget,image_auroc=auroc(labels,scores),
                    primary_auroc=auroc(labels,[r["score"] for r in dev_base]),
                    review_subset_secondary_auroc=auroc([labels[i] for i in review],[scores[i] for i in review]),
                    review_rate=len(review)/len(dev),secondary_call_fraction=len(review)/len(dev),
                    discovered_defects=sum(pool[i].label==1 for i in selected),
                    image_label_cost=fixed_labels+budget,initial_segmenter_normal_labels=len(initial),
                    development_evaluation_labels=len(dev),total_labels_including_development=fixed_labels+budget+len(dev),
                    pixel_mask_cost=sum(pool[i].label==1 and pool[i].mask is not None for i in selected),
                    annotation_seconds=None,training=model.training_record,
                    target_reached=bool(auroc(labels,scores)>=target))
                curve.append(row)
                dump_json(output/f"{stem}-ledger.json",ledger)
                dump_json(output/f"{stem}-dev-predictions.json",[dict(id=s.id,label=s.label,score=p) for s,p in zip(dev,scores)])
                dump_json(output/"curve.json",curve)
                print(f"AL {strategy} seed={seed} labels={budget} dev_AUROC={row['image_auroc']:.4f}",flush=True)
    summary = []
    for strategy in ("random","uncertainty","coreset","hybrid"):
        for budget in budgets:
            values = [r["image_auroc"] for r in curve if r["strategy"]==strategy and r["budget"]==budget]
            summary.append(dict(strategy=strategy,budget=budget,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0))
    reached = [dict(strategy=strategy,seed=seed,first_budget=next((r["budget"] for r in curve
        if r["strategy"]==strategy and r["seed"]==seed and r["target_reached"]),"not_reached"))
        for strategy in ("random","uncertainty","coreset","hybrid") for seed in seeds]
    report = dict(schema_version=2,manifest_sha256=fingerprint,base_policy_sha256=policy_hash(inspector.meta),
        seeds=list(seeds),budgets=budgets,steps_per_fit=steps,target_auroc=target,summary=summary,target_budgets=reached,
        protocol="Frozen primary; same-budget supervised U-Net; dev reused for research; no official test labels used",
        caveat="Advice-only cascade does not reduce live review rate. No labeling-time measurements; masks charged separately. Three seeds are not independent production datasets.")
    dump_json(output/"learning.json",report)
    return report
