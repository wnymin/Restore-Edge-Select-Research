"""Verify and materialize archived CPU lookahead audit tables."""

import argparse
from fnmatch import fnmatchcase
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/evidence'
ARCHIVE = EVIDENCE / 'lookahead-audit-csv.zip'
MANIFEST = EVIDENCE / 'lookahead-audit-csv.sha256'
CACHE = ROOT / 'outputs/lookahead-evidence'


@lru_cache(maxsize=1)
def manifest():
    hashes = {}
    for line in MANIFEST.read_text(encoding='ascii').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (dynamic-lookahead-[a-z0-9-]+\.csv)', line)
        if not match or match[2] in hashes:
            raise ValueError(f'Invalid or duplicate manifest entry: {line}')
        hashes[match[2]] = match[1]
    if not hashes:
        raise ValueError(f'Empty manifest: {MANIFEST}')
    return hashes


def checked_member(archive, name):
    data = archive.read(name)
    if sha256(data).hexdigest() != manifest()[name]:
        raise ValueError(f'Archived evidence hash mismatch: {name}')
    return data


def verify():
    with ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(manifest()):
            raise ValueError('Archive members do not match the manifest')
        return sum(len(checked_member(archive, name)) for name in manifest())


def csv_path(name):
    if not re.fullmatch(r'dynamic-lookahead-[a-z0-9-]+\.csv', name):
        raise ValueError(f'Expected a CPU lookahead CSV filename: {name}')
    if name not in manifest():
        path = EVIDENCE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    path = CACHE / name
    if path.is_file() and sha256(path.read_bytes()).hexdigest() == manifest()[name]:
        return path
    with ZipFile(ARCHIVE) as archive:
        data = checked_member(archive, name)
    CACHE.mkdir(parents=True, exist_ok=True)
    pending = CACHE / f'.{uuid4().hex}.tmp'
    try:
        pending.write_bytes(data)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path


def csv_paths(pattern):
    names = {path.name for path in EVIDENCE.glob(pattern)}
    names.update(name for name in manifest() if fnmatchcase(name, pattern))
    if not names:
        raise FileNotFoundError(f'No lookahead evidence matching {pattern}')
    return [csv_path(name) for name in sorted(names)]


def summary_path(path):
    directory = EVIDENCE if path.parent == CACHE else path.parent
    return directory / (path.stem + '-summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'verify', 'extract'))
    parser.add_argument('names', nargs='*')
    args = parser.parse_args()
    if args.names and args.action != 'extract':
        parser.error('names are only accepted for extract')
    if args.action == 'verify':
        size = verify()
        print(f'Verified {len(manifest())} CSV files ({size:,} bytes).')
    elif args.action == 'list':
        print('\n'.join(sorted(manifest())))
    else:
        for name in args.names or sorted(manifest()):
            print(csv_path(name).relative_to(ROOT))
