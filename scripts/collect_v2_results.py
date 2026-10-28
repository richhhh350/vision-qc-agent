"""Copy portable research evidence and render comparative figures/error examples."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vision_qc.data import load_manifest
from vision_qc.metrics import routing_metrics
from vision_qc.pipeline import dump_json, overlay
from vision_qc.v2.pipeline import InspectorV2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('data/ksdd2-v2.csv'))
    parser.add_argument('--output',type=Path,default=Path('results/v2'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    models=[]
    paths=[Path('runs/v2-patchcore'),Path('runs/v2-efficientad')]
    if Path('runs/v2-geometry/selected.json').exists():
        selected=json.loads(Path('runs/v2-geometry/selected.json').read_text(encoding='utf-8'))
        paths.append(Path('runs/v2-geometry')/f"candidate-{selected['candidate']}")
    for run in paths:
        if not (run/'metrics-test.json').exists():
            continue
        name=run.name if run.parent.name!='v2-geometry' else 'v2-geometry-selected'
        destination=args.output/name
        destination.mkdir(exist_ok=True)
        for file in run.glob('metrics-*.json'):
            report=json.loads(file.read_text(encoding='utf-8'))
            predictions_path=run/file.name.replace('metrics-','predictions-')
            if predictions_path.exists():
                predictions=json.loads(predictions_path.read_text(encoding='utf-8'))
                report['raw_conformal_routing']=routing_metrics([p['label'] for p in predictions],
                    [dict(p,route=p['raw_route']) for p in predictions])
            dump_json(destination/file.name,report)
        for pattern in ('predictions-*.json','config.json','onnx-parity.json','onnx-contract.json'):
            for file in run.glob(pattern):
                if file.name.startswith('predictions-'):
                    values=json.loads(file.read_text(encoding='utf-8'))
                    # Benchmark embeddings are not used for selection; keep the AL
                    # feature artifact separately and avoid huge per-test vectors.
                    dump_json(destination/file.name,[{k:v for k,v in row.items() if k!='embedding'} for row in values])
                else:
                    shutil.copy2(file,destination/file.name)
        meta=json.loads((run/'model.json').read_text(encoding='utf-8'))
        dump_json(destination/'model-receipt.json',{k:meta.get(k) for k in ('version','backend','weights','detector','counts','manifest_sha256','evidence','certificate','train_seconds')})
        metrics=json.loads((run/'metrics-test.json').read_text(encoding='utf-8'))
        models.append(dict(name=name,image_auroc=metrics['image_auroc'],pixel_auroc=metrics['pixel_auroc_binned'],
                           review_rate=metrics['review_rate'],latency_p95_ms=metrics['latency_p95_ms']))
    for source,name in [('docs/environment-v2-tested.json','environment.json'),('data/ksdd2-v2.protocol.json','data-protocol.json'),
                        ('runs/v2-geometry/dev-matrix.json','geometry-dev-matrix.json'),('runs/v2-geometry/preregistered.json','geometry-preregistered.json'),
                        ('runs/v2-http/benchmark.json','http-benchmark.json')]:
        if Path(source).exists():
            shutil.copy2(source,args.output/name)
    source=Path('runs/v2-learning-balanced')
    if (source/'learning.json').exists():
        original=Path('runs/v2-learning')
        if (original/'learning.json').exists():
            original_destination=args.output/'learning-uniform60'; original_destination.mkdir(exist_ok=True)
            for name in ('learning.json','curve.json'):
                shutil.copy2(original/name,original_destination/name)
        destination=args.output/'learning'; destination.mkdir(exist_ok=True)
        learning_report=json.loads((source/'learning.json').read_text(encoding='utf-8'))
        budgets=learning_report['budgets']
        # Include decisions, draw probabilities, revealed labels and predictions; exclude .pt files.
        for file in source.glob('*.json'):
            value=json.loads(file.read_text(encoding='utf-8'))
            if file.name.endswith('-ledger.json'):
                # Version identifies the model used BEFORE this batch was selected.
                strategy,seed,budget=file.name.removesuffix('-ledger.json').split('-')
                for i,row in enumerate(value):
                    round_index=next(j for j,b in enumerate(budgets) if i<b)
                    previous=budgets[round_index-1] if round_index else 0
                    if previous:
                        row['model_version']=f'{strategy}-{seed}-{previous}'
            dump_json(destination/file.name,value)
        summary=json.loads((source/'learning.json').read_text(encoding='utf-8'))['summary']
        fig,ax=plt.subplots(figsize=(7,4))
        for strategy in ('random','uncertainty','coreset','hybrid'):
            rows=[r for r in summary if r['strategy']==strategy]
            ax.errorbar([r['budget'] for r in rows],[r['mean'] for r in rows],yerr=[r['std'] for r in rows],marker='o',capsize=3,label=strategy)
        ax.set(xlabel='Additional image labels (pixel masks charged separately)',ylabel='Dev Image AUROC',ylim=(0,1),title='3-seed U-Net learning curves: mean ± sample SD')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(args.output/'learning-curves.png',dpi=160); plt.close(fig)
    if models:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        names=['v1']+[m['name'].removeprefix('v2-') for m in models]
        v1=json.loads(Path('results/v1/ksdd2-main.json').read_text(encoding='utf-8'))
        axes[0].bar(names,[v1['image_auroc']]+[m['image_auroc'] for m in models],color=['#94a3b8']+['#2766ad']*len(models))
        axes[0].set(ylim=(0,1),ylabel='Image AUROC',title='Repeated KSDD2 official test benchmark')
        axes[1].bar([m['name'].removeprefix('v2-') for m in models],[m['review_rate'] for m in models],color='#e4a142')
        axes[1].set(ylim=(0,1),ylabel='Review fraction',title='Release disabled: evidence insufficient')
        for ax in axes: ax.tick_params(axis='x',rotation=20)
        fig.tight_layout(); fig.savefig(args.output/'comparison.png',dpi=160); plt.close(fig)
        run=paths[0]; inspector=InspectorV2(run)
        predictions=json.loads((run/'predictions-test.json').read_text(encoding='utf-8'))
        samples={s.id:s for s in load_manifest(args.manifest)}
        hard=sorted([p for p in predictions if p['label']==1],key=lambda p:p['score'])[:2]+sorted([p for p in predictions if p['label']==0],key=lambda p:-p['score'])[:2]
        fig,axes=plt.subplots(2,4,figsize=(12,7))
        for column,row in enumerate(hard):
            with Image.open(samples[row['id']].path) as image:
                result,heatmap=inspector.inspect(image,samples[row['id']].category)
                axes[0,column].imshow(image)
                axes[1,column].imshow(overlay(image,heatmap,inspector.meta['pixel_threshold'],result['boxes']))
            axes[0,column].set_title(f"{row['id']}\nlabel={row['label']} score={row['score']:.3f}",fontsize=8)
            axes[1,column].set_title(f"raw={row['raw_route']}; service={row['route']}",fontsize=8)
        for ax in axes.flat: ax.axis('off')
        fig.suptitle('Lowest-scoring defects / highest-scoring normal images · KSDD2 CC BY-NC-SA 4.0')
        local_artifacts=Path('runs/v2-report-artifacts'); local_artifacts.mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig(local_artifacts/'hard-cases.png',dpi=140); plt.close(fig)
    files=sorted(Path('src/vision_qc').rglob('*.py'))+sorted(Path('scripts').glob('*.py'))+sorted(Path('tests').glob('*.py'))+sorted(Path('configs/v2').glob('*.json'))+[Path('dashboard.py'),Path('pyproject.toml'),Path('requirements-v2-tested.txt')]
    files=sorted(files,key=lambda p:p.as_posix())
    code_hash=hashlib.sha256(b''.join(p.as_posix().encode()+p.read_bytes() for p in files)).hexdigest()
    dump_json(args.output/'summary.json',dict(models=models,source_sha256=code_hash,
        git_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        note='Working-tree source hash includes unpublished v2 edits; full public test is a repeated benchmark'))
    print(json.dumps(models,indent=2))


if __name__=='__main__':
    main()
