"""Run the unchanged sealed R08 driver using its exact historical simulator."""
import gzip
from hashlib import sha256
import runpy

from scripts.validation.archive_source import ROOT, R08_SIMULATOR_SHA256


def main():
    # Import before importing any R08 modules that bind Experiment.
    from restore_edge_select import simulator
    raw = gzip.decompress((ROOT/'docs/evidence/r08-frozen-simulator.py.gz').read_bytes())
    if sha256(raw).hexdigest() != R08_SIMULATOR_SHA256:
        raise ValueError('Historical simulator source pin mismatch')
    exec(compile(raw,simulator.__file__,'exec'),simulator.__dict__)
    simulator._archive_r08_verified_source = R08_SIMULATOR_SHA256
    runpy.run_module('scripts.validation.run_r08_sealed_holdout',run_name='__main__')


if __name__=='__main__':
    main()
