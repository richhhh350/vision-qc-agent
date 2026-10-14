"""Train/evaluate predeclared configs on dev; freeze winner before official test."""
import argparse
import json
from pathlib import Path

from vision_qc.pipeline import dump_json
from vision_qc.v2.pipeline import train_v2, evaluate_v2
from vision_qc.v2.evidence import hash_json, policy_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--configs',type=Path,nargs='+',default=[Path('configs/v2/patchcore-letterbox256.json'),Path('configs/v2/patchcore-letterbox384.json')])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    configs=[json.loads(p.read_text(encoding='utf-8')) for p in args.configs]
    dump_json(args.output/'preregistered.json',dict(configs=configs,selection='maximum dev image AUROC, tie lower dev latency',
        official_test='Only selected candidate; repeated public benchmark, not production evidence'))
    rows=[]
    for i,config in enumerate(configs):
        run=args.output/f'candidate-{i}'
        meta=train_v2(args.manifest,run,config)
        report=evaluate_v2(run,args.manifest,'dev')
        rows.append(dict(candidate=i,config_sha256=hash_json(config),policy_sha256=policy_hash(meta),
                         image_auroc=report['image_auroc'],latency_p95_ms=report['latency_p95_ms'],
                         geometry=config['detector']['geometry']))
        dump_json(args.output/'dev-matrix.json',rows)
    selected=max(rows,key=lambda r:(r['image_auroc'],-r['latency_p95_ms']))
    dump_json(args.output/'selected.json',selected)
    evaluate_v2(args.output/f"candidate-{selected['candidate']}",args.manifest,'test')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':
    main()
