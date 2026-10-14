"""Record the tested dependency closure without copying unrelated global packages."""
from pathlib import Path
import importlib.metadata as metadata
import json
import os
import platform
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    roots=['anomalib','torch','torchvision','timm','lightning','numpy','scipy','Pillow','fastapi','uvicorn',
           'python-multipart','streamlit','httpx','pytest','onnx','onnxruntime','matplotlib','psutil']
    pending=list(roots); versions={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in versions:
            continue
        dist=metadata.distribution(name)
        versions[name]=dist.version
        for raw in dist.requires or []:
            req=Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra':''}):
                pending.append(req.name)
    Path('requirements-v2-tested.txt').write_text('# Tested Windows Python 3.14 dependency closure; CPU execution.\n'+
        '\n'.join(f'{name}=={version}' for name,version in sorted(versions.items()))+'\n',encoding='utf-8')
    environment=dict(python=platform.python_version(),platform=platform.platform(),processor=os.getenv('PROCESSOR_IDENTIFIER'),
        packages=versions,installation='Current .venv-v2 inherits base site-packages; setup-v2.ps1 creates an isolated environment for fresh installs',
        package_source='PyPI; timm pretrained weights from Hugging Face; EfficientAD teacher from official Anomalib release')
    Path('docs/environment-v2-tested.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print(f'Recorded {len(versions)} packages')


if __name__=='__main__':
    main()
