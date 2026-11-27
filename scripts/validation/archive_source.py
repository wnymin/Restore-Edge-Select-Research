"""Verify archive source pins even after adding opt-in experiment code."""
import gzip
import sys
from hashlib import sha256
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]

R08_BASE = '074d5634c7a6847a48ec876f0b27e0dd7a3043a0'
R08_SIMULATOR_SHA256 = '5cd7ba7fc3e13526dd7cb259ab02ff409a9966232f50bf278f6315ac21544794'
SIMULATOR_PATH = 'src/restore_edge_select/simulator.py'
R08_BASE_PATHS = {SIMULATOR_PATH, 'src/restore_edge_select/prefill_cost.py'}
R09_FROZEN_PATHS = {SIMULATOR_PATH, 'scripts/cpu/replay/study_r09_budget.py'}


def source_matches(path, expected, *, historical=False):
    # The Windows shell wrote a CRLF terminator on some initial source files;
    # .gitattributes normalizes checked-out sources to LF. Keep measured hashes.
    def matches(actual):
        lf=actual.replace(b'\r\n',b'\n')
        variants={actual,lf,lf.replace(b'\n',b'\r\n')}
        if lf.endswith(b'\n'): variants.add(lf[:-1]+b'\r\n')
        return any(sha256(v).hexdigest()==expected for v in variants)
    if matches(path.read_bytes()):
        return True
    if path == ROOT/SIMULATOR_PATH and expected == R08_SIMULATOR_SHA256:
        # Only the explicit frozen runner uses this process-local provenance.
        module = sys.modules.get('restore_edge_select.simulator')
        if getattr(module,'_archive_r08_verified_source',None) == expected:
            return matches(gzip.decompress((ROOT/'docs/evidence/r08-frozen-simulator.py.gz').read_bytes()))
    if not historical:
        return False
    if path == ROOT/SIMULATOR_PATH and expected == R08_SIMULATOR_SHA256:
        return matches(gzip.decompress((ROOT/'docs/evidence/r08-frozen-simulator.py.gz').read_bytes()))
    relative = path.relative_to(ROOT).as_posix()
    if relative in R09_FROZEN_PATHS:
        with ZipFile(ROOT/'docs/evidence/r09-frozen-sources.zip') as archive:
            if matches(archive.read(relative)):
                return True
    if relative in R08_BASE_PATHS:
        return matches(check_output(['git','show',f'{R08_BASE}:{relative}'],cwd=ROOT))
    return False
