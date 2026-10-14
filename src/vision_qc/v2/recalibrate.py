"""Fresh calibration and single-use acceptance for an already frozen model package."""
import json
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from PIL import Image

from ..calibration import ConformalRouter, ScoreProbability
from ..data import load_manifest
from ..metrics import routing_metrics
from ..pipeline import dump_json
from .evidence import fingerprints, hash_json, policy_hash, certificate, check_certificate
from .pipeline import InspectorV2


def recalibrate_v2(run, manifest, output, *, independent=False):
    run,manifest,output=Path(run),Path(manifest),Path(output)
    if output.exists():
        raise FileExistsError(output)
    inspector=InspectorV2(run)
    samples=load_manifest(manifest)
    if {s.category for s in samples}!={inspector.meta['category']}:
        raise ValueError('Recalibration category mismatch')
    if any(s.split not in {'prob_fit','calibration','acceptance'} for s in samples):
        raise ValueError('Fresh manifest may contain only prob_fit/calibration/acceptance')
    calibration=[s for s in samples if s.split=='calibration']
    acceptance=[s for s in samples if s.split=='acceptance']
    if not calibration or not acceptance:
        raise ValueError('Fresh calibration and acceptance splits required')
    if len({s.group for s in samples})!=len(samples):
        raise ValueError('One image per independent group required')
    records,_=fingerprints(samples)
    history=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    historical_pixels={r['pixels'] for r in history}
    historical_groups={r['group'] for r in history}
    if any(r['pixels'] in historical_pixels or r['group'] in historical_groups for r in records):
        raise ValueError('Fresh evidence overlaps historical images/groups')
    # A local append-only usage ledger prevents repeated acceptance against sibling
    # candidates with the same initial model. Reserved before reading predictions.
    ledger=run.parent/'acceptance-usage.sqlite'
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE IF NOT EXISTS used(pixels TEXT PRIMARY KEY,group_id TEXT UNIQUE,at TEXT NOT NULL,run TEXT NOT NULL)')
        try:
            db.execute('BEGIN IMMEDIATE')
            db.executemany('INSERT INTO used VALUES(?,?,?,?)',[(r['pixels'],r['group'],datetime.now(timezone.utc).isoformat(),str(output.resolve())) for r in records])
        except sqlite3.IntegrityError:
            raise ValueError('Evidence already reserved by an earlier calibration/acceptance attempt') from None
    meta=json.loads(json.dumps(inspector.meta))
    scored={}
    for split in ('prob_fit','calibration'):
        scored[split]=[]
        for s in samples:
            if s.split==split:
                with Image.open(s.path) as image:
                    scored[split].append((s.label,inspector.model.predict(image).score))
    if scored['prob_fit']:
        meta['probability']=ScoreProbability.fit([v for _,v in scored['prob_fit']],[l for l,_ in scored['prob_fit']]).to_dict()
    meta['conformal']=ConformalRouter([v for l,v in scored['calibration'] if l==0],
        [v for l,v in scored['calibration'] if l==1],meta['conformal']['alpha_normal'],meta['conformal']['alpha_defect']).to_dict()
    meta['parent_version']=meta['version']; meta['version']='v2-'+policy_hash(meta)[:16]
    meta['certificate']=None
    meta['evidence']=dict(calibration_sha256=hash_json([r for r in records if r['split']=='calibration']),
                          acceptance_sha256=hash_json([r for r in records if r['split']=='acceptance']))
    output.mkdir(parents=True)
    for name in meta['weights']:
        shutil.copy2(run/name,output/name)
    for name in ('feedback-snapshot.json','config.json'):
        if (run/name).exists():
            shutil.copy2(run/name,output/name)
    dump_json(output/'provenance.json',history+records)
    dump_json(output/'model.json',meta)
    # Policy is written/frozen before any acceptance predictions.
    inspector=InspectorV2(output)
    decisions=[]
    for s in acceptance:
        with Image.open(s.path) as image:
            decisions.append(inspector.inspect(image,s.category,enforce_gate=False)[0])
    metrics=routing_metrics([s.label for s in acceptance],decisions)
    meta['certificate']=certificate(meta,metrics,meta['evidence']['calibration_sha256'],meta['evidence']['acceptance_sha256'],independent=independent)
    approved,reason=check_certificate(meta)
    meta['release_gate'].update(approved=approved,reason=reason,validation=metrics)
    meta['counts'].update({split:sum(s.split==split for s in samples) for split in ('prob_fit','calibration','acceptance')})
    dump_json(output/'model.json',meta)
    dump_json(output/'acceptance-predictions.json',[dict(id=s.id,label=s.label,**d) for s,d in zip(acceptance,decisions)])
    return dict(version=meta['version'],release_gate=meta['release_gate'],certificate=meta['certificate'])
