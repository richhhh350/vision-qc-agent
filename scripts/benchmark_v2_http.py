"""Isolated batch=1 HTTP benchmark, with decoding, rendering and SQLite included."""
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from collections import Counter

import httpx
import numpy as np
import psutil
from PIL import Image
from vision_qc.data import load_manifest
from vision_qc.pipeline import dump_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,default=200)
    args=parser.parse_args()
    if args.requests<200:
        raise ValueError('At least 200 measured requests required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    samples=[s for s in load_manifest(args.manifest) if s.split=='dev']
    if not samples:
        raise ValueError('Nonempty dev split required')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    env=dict(os.environ,QC_RUN_DIR=str(args.run.resolve()),QC_STATE_DIR=str((args.output/'state').resolve()),PYTHONUTF8='1')
    env.pop('QC_REGISTRY_DIR',None)
    flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
    log=(args.output/'server.log').open('w',encoding='utf-8')
    server=subprocess.Popen([sys.executable,'-m','uvicorn','vision_qc.api:create_app','--factory','--host','127.0.0.1','--port',str(port)],
                            env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    try:
        latencies=[]; stages=[]; rss=[]; sizes=[]; labels=[]; decisions=[]
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=120,headers={'X-API-Key':env.get('QC_API_KEY','')}) as client:
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError('Benchmark server failed; see server.log')
                try:
                    if client.get('/health').status_code==200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Server readiness timeout')
            for i in range(args.requests+10):
                sample=samples[i%len(samples)]
                raw=sample.path.read_bytes()
                with Image.open(sample.path) as image:
                    image_size=f'{image.width}x{image.height}'
                start=time.perf_counter()
                response=client.post('/inspect',files={'file':('image.png',raw)},data={'category':sample.category})
                response.raise_for_status()
                elapsed=(time.perf_counter()-start)*1000
                if i>=10:
                    latencies.append(elapsed)
                    stages.append(response.json()['result'].get('stage_timing',{}))
                    sizes.append(image_size); labels.append(sample.label); decisions.append(response.json()['result'])
                    processes=[psutil.Process(server.pid)]+psutil.Process(server.pid).children(recursive=True)
                    rss.append(sum(p.memory_info().rss for p in processes if p.is_running()))
                if (i+1)%50==0:
                    print(f'HTTP {i+1}/{args.requests+10}',flush=True)
        meta=json.loads((args.run/'model.json').read_text(encoding='utf-8'))
        keys={k for row in stages for k in row if k.endswith('_ms')}
        cpu=os.getenv('PROCESSOR_IDENTIFIER',platform.processor())
        if sys.platform=='win32':
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                    cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
            except OSError:
                pass
        report=dict(n=args.requests,warmup=10,batch=1,concurrency=1,
            model_version=meta['version'],backend=meta['backend'],geometry=meta['detector'].get('geometry'),threads=meta['detector'].get('threads'),
            http_p50_ms=float(np.percentile(latencies,50)),http_p95_ms=float(np.percentile(latencies,95)),
            throughput_images_per_second=1000/float(np.mean(latencies)),
            peak_sampled_process_rss_bytes=max(rss),rss_protocol='sum server and child RSS sampled after each request; not OS peak',
            model_bytes=sum((args.run/name).stat().st_size for name in meta['weights']),
            stages_p95_ms={key:float(np.percentile([s.get(key,0) for s in stages],95)) for key in keys},
            cpu=cpu,logical_cpus=psutil.cpu_count(),input_dimensions=dict(Counter(sizes)),labels=dict(Counter(labels)),
            secondary_call_fraction=sum(r.get('secondary') is not None for r in decisions)/len(decisions),
            routes=dict(Counter(r['route'] for r in decisions)),
            platform=platform.platform(),python=platform.python_version(),
            scope='localhost HTTP including multipart, decoding, inference, overlay PNG, SQLite and JSON response; client file read excluded',
            latency_target_ms=250,target_met=float(np.percentile(latencies,95))<=250)
        dump_json(args.output/'benchmark.json',report)
        print(json.dumps(report,indent=2))
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        log.close()


if __name__=='__main__':
    main()
