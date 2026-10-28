"""V2 training, evaluation, export, registry and learning commands."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('train')
    fit.add_argument('--manifest',type=Path,required=True)
    fit.add_argument('--output',type=Path,required=True)
    fit.add_argument('--config',type=Path,default=Path('configs/v2/patchcore-cpu.json'))
    test=sub.add_parser('evaluate')
    test.add_argument('--run',type=Path,required=True)
    test.add_argument('--manifest',type=Path,required=True)
    test.add_argument('--split',default='test',choices=['dev','acceptance','test'])
    export=sub.add_parser('export')
    export.add_argument('--run',type=Path,required=True)
    export.add_argument('--manifest',type=Path,required=True)
    export.add_argument('--verify-existing',action='store_true')
    learn=sub.add_parser('learn')
    learn.add_argument('--run',type=Path,required=True)
    learn.add_argument('--manifest',type=Path,required=True)
    learn.add_argument('--output',type=Path,required=True)
    learn.add_argument('--budgets',type=int,nargs='+',default=[10,20,40])
    learn.add_argument('--steps',type=int,default=200)
    learn.add_argument('--seeds',type=int,nargs='+',default=[17,42,2026])
    feedback=sub.add_parser('retrain')
    for name in ('run','state','output'):
        feedback.add_argument('--'+name,type=Path,required=True)
    feedback.add_argument('--steps',type=int,default=120)
    recalibrate=sub.add_parser('recalibrate')
    for name in ('run','manifest','output'):
        recalibrate.add_argument('--'+name,type=Path,required=True)
    recalibrate.add_argument('--independent-production-sampling',action='store_true',
                            help='Explicitly attest this is fresh representative independent production evidence, not public benchmark reuse')
    registry=sub.add_parser('registry')
    registry.add_argument('action',choices=['register','activate','rollback','list'])
    registry.add_argument('--directory',type=Path,default=Path('runs/registry'))
    registry.add_argument('--run',type=Path)
    registry.add_argument('--version')
    registry.add_argument('--require-release',action='store_true')
    a=parser.parse_args()
    if a.command == 'train':
        from vision_qc.v2.pipeline import train_v2
        result=train_v2(a.manifest,a.output,json.loads(a.config.read_text(encoding='utf-8')))
    elif a.command == 'evaluate':
        from vision_qc.v2.pipeline import evaluate_v2
        result=evaluate_v2(a.run,a.manifest,a.split)
    elif a.command == 'export':
        from vision_qc.v2.export import export_v2
        result=export_v2(a.run,a.manifest,verify_only=a.verify_existing)
    elif a.command == 'learn':
        from vision_qc.v2.learning import run_learning
        result=run_learning(a.run,a.manifest,a.output,budgets=a.budgets,seeds=a.seeds,steps=a.steps)
    elif a.command == 'retrain':
        from vision_qc.v2.feedback import retrain_feedback
        result=retrain_feedback(a.run,a.state,a.output,a.steps)
    elif a.command == 'recalibrate':
        from vision_qc.v2.recalibrate import recalibrate_v2
        result=recalibrate_v2(a.run,a.manifest,a.output,independent=a.independent_production_sampling)
    else:
        from vision_qc.v2.registry import Registry
        registry=Registry(a.directory)
        if a.action=='register':
            if not a.run:
                parser.error('--run required')
            result=registry.register(a.run)
        elif a.action=='activate':
            if not a.version:
                parser.error('--version required')
            result=registry.activate(a.version,require_release=a.require_release)
        elif a.action=='rollback':
            result=registry.rollback()
        else:
            result=registry.list()
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
