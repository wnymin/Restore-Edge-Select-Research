"""Fetch pinned Bailian files into data/raw/bailian and verify Git LFS hashes."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def valid(path, entry):
    if not path.exists() or path.stat().st_size != entry["bytes"]:
        return False
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest() == entry["sha256"]


def fetch(url, target, entry=None):
    if entry is not None and valid(target, entry):
        print(f"Verified {target.name}")
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(url, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    stream.write(chunk)
            stream.close()
            if entry is not None and not valid(temporary, entry):
                raise ValueError(f"Size or SHA-256 mismatch: {target.name}")
            temporary.replace(target)
            print(f"Saved {target.name}")
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/bailian-manifest.json").read_text(encoding="utf-8"))
    directory = ROOT / "data/raw/bailian"
    base = manifest["repository"] + "/" + manifest["revision"]
    if args.verify_only:
        invalid = []
        for entry in manifest["files"]:
            if valid(directory / entry["name"], entry):
                print(f"Verified {entry['name']}")
            else:
                invalid.append(entry["name"])
        if invalid:
            parser.error("Missing or mismatched: " + ", ".join(invalid))
        return
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for entry in manifest["files"]:
            fetch(f"https://github.com/{manifest['repository']}/raw/{manifest['revision']}/{entry['name']}",
                  directory / entry["name"], entry)
        fetch(f"https://raw.githubusercontent.com/{base}/LICENSE", directory / "LICENSE")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
