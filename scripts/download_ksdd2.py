"""Download the public dataset from ViCoS Lab and record its local SHA-256.

Official source: https://www.vicos.si/resources/kolektorsdd2/
License: CC BY-NC-SA 4.0 (per the official page, checked 2026-10-13).
"""
import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/public/ksdd2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "KolektorSDD2.zip"
    if not archive.exists():
        part = archive.with_suffix(".zip.part")
        request = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("Server does not support range downloads")
            total = int(response.headers["Content-Range"].split("/")[-1])
        if total > 2 * 1024**3:
            raise ValueError("Unexpected archive size")
        chunks = args.output / "download_chunks"
        chunks.mkdir(exist_ok=True)
        chunk_size = 8 * 1024**2
        # Preserve fully downloaded chunks from an interrupted sequential transfer.
        if part.exists():
            with part.open("rb") as previous:
                for index in range(part.stat().st_size // chunk_size):
                    chunk = previous.read(chunk_size)
                    destination = chunks / f"{index:04d}.part"
                    if not destination.exists():
                        destination.write_bytes(chunk)
        def fetch(index):
            start, end = index * chunk_size, min((index + 1) * chunk_size, total) - 1
            destination = chunks / f"{index:04d}.part"
            if destination.exists() and destination.stat().st_size == end - start + 1:
                return end - start + 1
            for attempt in range(3):
                try:
                    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                            raise ValueError("Range mismatch")
                        block = response.read()
                    if len(block) != end - start + 1:
                        raise ValueError("Incomplete chunk")
                    destination.write_bytes(block)
                    return len(block)
                except Exception:
                    if attempt == 2:
                        raise
        count, received = (total + chunk_size - 1) // chunk_size, 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, index) for index in range(count)]
            for future in as_completed(futures):
                received += future.result()
                print(f"Verified chunks {received // 1024**2} / {total // 1024**2} MiB", flush=True)
        with part.open("wb") as stream:
            for index in range(count):
                stream.write((chunks / f"{index:04d}.part").read_bytes())
        part.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    target = (args.output / "dataset").resolve()
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if sum(info.file_size for info in z.infolist()) > 4 * 1024**3:
            raise ValueError("Unexpected uncompressed size")
        for info in z.infolist():
            resolved = (target / info.filename).resolve()
            if not resolved.is_relative_to(target) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe archive path")
        z.extractall(target)
    provenance = dict(source=URL, official_page="https://www.vicos.si/resources/kolektorsdd2/",
                      license="CC BY-NC-SA 4.0", archive_sha256=digest,
                      checksum_note="Locally computed fingerprint; publisher checksum was not provided",
                      citation="Bozic, Tabernik, Skocaj. Mixed supervision for surface-defect detection: from weakly to fully supervised learning. Computers in Industry, 2021.")
    (args.output / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(dict(extracted_to=str(target), **provenance), indent=2), flush=True)


if __name__ == "__main__":
    main()
