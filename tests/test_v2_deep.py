"""Offline architecture/serialization checks; never claimed as accuracy evidence."""
import os

import numpy as np
from PIL import Image
import pytest

pytestmark = pytest.mark.skipif(os.getenv("QC_RUN_DEEP_TESTS")!="1",reason="Opt-in deep CPU dependencies")


def test_patchcore_official_fit_load_contract(tmp_path):
    pytest.importorskip("anomalib")
    from vision_qc.v2.detectors import DeepDetector, load_detector
    from vision_qc.data import synthetic_image
    images=[]
    for i in range(4):
        image,_=synthetic_image(np.random.default_rng(i),False)
        path=tmp_path/f"{i}.png"; image.save(path); images.append(path)
    detector=DeepDetector(geometry=dict(size=64,mode="tiles"),bank_size=16,max_features=128,threads=2,pretrained=False)
    detector.fit(images)
    assert detector.network.memory_bank.shape[0]==16
    detector.save(tmp_path)
    loaded=load_detector(tmp_path,detector.metadata())
    for mode in ("L","RGBA"):
        image=Image.new(mode,(77,113))
        a,b=detector.predict(image),loaded.predict(image)
        assert a.heatmap.shape==(113,77) and np.isfinite(a.heatmap).all()
        assert np.isclose(a.score,b.score) and np.allclose(a.heatmap,b.heatmap)
        assert np.isfinite(a.embedding).all()


def test_efficientad_contract_and_save(tmp_path):
    pytest.importorskip("anomalib")
    from vision_qc.v2.detectors import DeepDetector, load_detector
    detector=DeepDetector(backend="efficientad",threads=2,pretrained=False)
    detector.save(tmp_path)
    loaded=load_detector(tmp_path,detector.metadata())
    image=Image.new("L",(101,213),128)
    a,b=detector.predict(image),loaded.predict(image)
    assert a.heatmap.shape==(213,101) and np.isfinite(a.heatmap).all()
    assert np.isclose(a.score,b.score)
