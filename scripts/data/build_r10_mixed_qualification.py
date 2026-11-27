"""Build small synthetic, variable-catalogue fixtures for native state qualification."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from scripts.data.build_r10_native_plans import digest, write_plan


def cases():
    rand = Random(102761)
    prompt = lambda n: [rand.randint(300, 50000) for _ in range(n)]
    common = prompt(1024)

    def case(ident, catalog):
        requests = [dict(id=f'p{i}', tokens=tokens, arrival_ms=str(i), boundaries=[0, *bounds])
                    for i, (tokens, bounds) in enumerate(catalog)]
        fingerprint = sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
        return dict(id=ident, family='synthetic', split='calibration', source_sha256=fingerprint, requests=requests)

    return [
        case('mixed-private-staged', [(prompt(2304), []), (common + prompt(128), [1024]), (common[:512] + prompt(31), [512])]),
        case('direct-handoff', [(common[:512] + prompt(31), [512]), (common[:512] + prompt(87), [512])]),
        case('nonadjacent-loan', [(common[:512] + prompt(76), [512]), (prompt(288), []), (common[:512] + prompt(19), [512])]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [write_plan(b, a.out) for b in cases()]
    with a.manifest.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(scope='synthetic event-anchored correctness only', generator_sha256=digest(Path(__file__)), blocks=rows),
                         ensure_ascii=False, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
