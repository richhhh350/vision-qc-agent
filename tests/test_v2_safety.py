import copy
import json
import shutil

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from vision_qc.api import create_app
from vision_qc.data import write_manifest
from vision_qc.v2.evidence import certificate
from vision_qc.v2.pipeline import train_v2
from vision_qc.metrics import binomial_upper


def small_run(tmp_path):
    from vision_qc.data import synthetic_image
    rng=np.random.default_rng(678)
    rows=[]
    for split in ('train','prob_fit','calibration','acceptance'):
        for i in range(4):
            image,_=synthetic_image(rng,split!='train' and i>=2)
            path=tmp_path/f'{split}-{i}.png'; image.save(path)
            rows.append(dict(id=path.stem,path=str(path),category='test',split=split,label=int(split!='train' and i>=2),group=path.stem,mask=''))
    write_manifest(tmp_path/'manifest.csv',rows)
    train_v2(tmp_path/'manifest.csv',tmp_path/'run',dict(detector=dict(backend='patch_memory_cpu',bank_size=16)))
    return tmp_path/'run'


def test_expired_certificate_cannot_replay_cached_pass(tmp_path,monkeypatch):
    run=small_run(tmp_path)
    meta=json.loads((run/'model.json').read_text(encoding='utf-8'))
    meta['conformal']['alpha_defect']=.4
    meta['signature_low']=[-1,-1,-1]; meta['signature_high']=[2,2,2]
    meta['audit_fraction']=0
    meta['certificate']=certificate(meta,dict(n_defect=300,missed_defects=0,fnr_upper_95=binomial_upper(0,300),false_alarm_rate=0),
        meta['evidence']['calibration_sha256'],meta['evidence']['acceptance_sha256'],independent=True)
    (run/'model.json').write_text(json.dumps(meta),encoding='utf-8')
    client=TestClient(create_app(run,tmp_path/'state'))
    raw=(tmp_path/'train-0.png').read_bytes()
    first=client.post('/inspect',files={'file':('x.png',raw)},data={'category':'test'}).json()
    assert first['result']['route']=='pass'
    monkeypatch.setattr('vision_qc.v2.pipeline.check_certificate',lambda meta:(False,'certificate_expired'))
    second=client.post('/inspect',files={'file':('x.png',raw)},data={'category':'test'}).json()
    assert first['id']==second['id']
    assert second['result']['route']=='review' and 'certificate_expired' in second['result']['reason']


def test_repeated_acceptance_group_is_rejected(tmp_path):
    import csv
    run=small_run(tmp_path)
    with (tmp_path/'manifest.csv').open() as file:
        rows=list(csv.DictReader(file))
    for row in rows:
        if row['split']=='acceptance': row['group']='same-piece'
    write_manifest(tmp_path/'grouped.csv',rows)
    with pytest.raises(ValueError,match='independent group'):
        train_v2(tmp_path/'grouped.csv',tmp_path/'invalid',dict(detector=dict(backend='patch_memory_cpu',bank_size=16)))


def test_live_normal_and_pixel_feedback_train_isolated_candidate(tmp_path):
    pytest.importorskip('torch')
    import io
    from vision_qc.data import synthetic_image
    from vision_qc.v2.feedback import retrain_feedback
    run=small_run(tmp_path)
    before=(run/'model.json').read_bytes()
    state=tmp_path/'state'; client=TestClient(create_app(run,state))
    for label in (0,1):
        image,mask=synthetic_image(np.random.default_rng(1234+label),bool(label))
        raw=io.BytesIO(); image.save(raw,format='PNG')
        row=client.post('/inspect',files={'file':('x.png',raw.getvalue())},data={'category':'test'}).json()
        url=f"/reviews/{row['id']}"
        assert client.post(url+'/feedback',json=dict(label=label,reviewer='test',revision=0)).status_code==200
        if label:
            raw=io.BytesIO(); mask.save(raw,format='PNG')
            assert client.post(url+'/mask',files={'file':('m.png',raw.getvalue())},data={'revision':1}).status_code==200
    result=retrain_feedback(run,state,tmp_path/'candidate',steps=2)
    assert result['normal_labels']==1 and result['defect_masks']==1
    assert result['release_approved'] is False
    assert (run/'model.json').read_bytes()==before
    meta=json.loads((tmp_path/'candidate'/'model.json').read_text(encoding='utf-8'))
    assert meta['secondary']['training']['defect_masks']==1
    assert meta['counts']['train']==5


def test_interrupted_job_state_is_persistent(tmp_path):
    from vision_qc.v2.jobs import TrainingJobs
    root=tmp_path/'jobs'; root.mkdir()
    identifier='a'*32
    (root/f'{identifier}.json').write_text(json.dumps(dict(id=identifier,status='running')),encoding='utf-8')
    jobs=TrainingJobs(tmp_path/'model',tmp_path)
    assert jobs.get(identifier)['status']=='interrupted'
    assert jobs.cancel(identifier)['status']=='interrupted'
    with pytest.raises(KeyError):
        jobs.get('../not-a-job')


def test_fresh_recalibration_rejects_history_and_repeated_attempt(tmp_path):
    import csv
    from vision_qc.data import synthetic_image
    from vision_qc.v2.recalibrate import recalibrate_v2
    run=small_run(tmp_path)
    with (tmp_path/'manifest.csv').open() as file:
        reused=[r for r in csv.DictReader(file) if r['split'] in {'calibration','acceptance'}]
    write_manifest(tmp_path/'reused.csv',reused)
    with pytest.raises(ValueError,match='overlaps'):
        recalibrate_v2(run,tmp_path/'reused.csv',tmp_path/'bad')
    rows=[]; rng=np.random.default_rng(9891)
    for split in ('calibration','acceptance'):
        for i in range(6):
            image,_=synthetic_image(rng,bool(i%2))
            path=tmp_path/f'fresh-{split}-{i}.png'; image.save(path)
            rows.append(dict(id=path.stem,path=str(path),category='test',split=split,label=i%2,group=path.stem,mask=''))
    write_manifest(tmp_path/'fresh.csv',rows)
    result=recalibrate_v2(run,tmp_path/'fresh.csv',tmp_path/'recalibrated')
    assert not result['release_gate']['approved']
    assert result['certificate']['n_defect']==3
    with pytest.raises(ValueError,match='reserved'):
        recalibrate_v2(run,tmp_path/'fresh.csv',tmp_path/'repeat')
