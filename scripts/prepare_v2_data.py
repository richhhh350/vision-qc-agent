"""Fixed development protocol, using ONLY original training images. Test remains intact."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from vision_qc.data import load_manifest, write_manifest
from vision_qc.pipeline import dump_json
from vision_qc.v2.evidence import fingerprints


def prepare(source, output):
    if output.exists():
        raise FileExistsError(output)
    samples = load_manifest(source)
    rng = np.random.default_rng(42)
    rows = []
    for label in (0, 1):
        available = [s for s in samples if s.split in {"train", "pool"} and s.label == label]
        available = [available[int(i)] for i in rng.permutation(len(available))]
        counts = ([len(available)//2]+[len(available)//10]*4) if label == 0 else [0,20,20,99,50]
        if sum(counts) > len(available):
            raise ValueError("This fixed KSDD2 protocol requires >=189 training defects")
        counts.append(len(available)-sum(counts))
        offset = 0
        for split, count in zip(["train","dev","prob_fit","calibration","acceptance","pool"], counts):
            for s in available[offset:offset+count]:
                rows.append(dict(id=s.id,path=str(s.path),category=s.category,split=split,label=label,
                                 group=s.group,mask=str(s.mask) if s.mask else ""))
            offset += count
    for s in samples:
        if s.split == "test":
            rows.append(dict(id=s.id,path=str(s.path),category=s.category,split="test",label=s.label,
                             group=s.group,mask=str(s.mask) if s.mask else ""))
    write_manifest(output,rows)
    records, digest = fingerprints(load_manifest(output))
    dump_json(output.with_suffix('.protocol.json'),dict(schema_version=2,seed=42,
        counts=dict(Counter(f'{r["split"]}:{r["label"]}' for r in rows)),portable_fingerprint=digest,
        official_test_preserved=True,independent_production_acceptance=False,
        caveat="Original train reused for research; only 50 acceptance defects, 1%/95% certification impossible even at zero misses"))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/ksdd2-official.csv'))
    p.add_argument('--output',type=Path,default=Path('data/ksdd2-v2.csv'))
    a=p.parse_args()
    prepare(a.source,a.output)
