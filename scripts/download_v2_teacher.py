"""Download only the official EfficientAD small teacher, with local provenance."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def main():
    target = Path("data/pretrained/efficientad/pretrained_teacher_small.pth")
    url = "https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
    if target.exists():
        print(f"Already exists: {target}")
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = [name for name in bundle.namelist() if Path(name).name == target.name]
        if len(names) != 1:
            raise ValueError("Expected one small teacher in official archive")
        weights = bundle.read(names[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(weights)
    target.with_suffix(".source.json").write_text(json.dumps(dict(url=url,
        archive_sha256=hashlib.sha256(archive).hexdigest(), sha256=hashlib.sha256(weights).hexdigest(),
        note="Local fingerprint; not a publisher-supplied checksum"), indent=2), encoding="utf-8")
    print(f"Saved {len(weights)} bytes: {target}")


if __name__ == "__main__":
    main()
