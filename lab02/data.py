"""Real S-parameter measurements -> gamma using the unchanged author algorithm."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OFFSETS_MM = (0, 21, 66, 81, 84, 93, 117, 123, 171, 192)


def verify_sources():
    manifest = json.loads((ROOT/'sources.json').read_text())
    for rec in manifest['files']:
        path = ROOT/rec['path']
        if path.stat().st_size != rec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != rec['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
    return len(manifest['files'])


def extract_measurements():
    import skrf as rf
    path = ROOT/'vendor/hatab/multinetwork.py'
    spec = importlib.util.spec_from_file_location('lab02_hatab', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = {}
    for name in ('ENA', 'ZNA', 'VectorStar'):
        networks = [rf.Network(str(ROOT/'data/measurements'/name/f'line_{d:03d}mm.s2p'))['3ghz-18ghz']
                    for d in OFFSETS_MM]
        f = networks[0].f
        s = np.stack([net.s for net in networks])
        er, kappa = 1-1e-6j, -1+0j
        values = []
        for i, frequency in enumerate(f):
            value, er, kappa, _ = module.mN_at_one_freq(
                s[:, i], np.array(OFFSETS_MM)*1e-3, er, kappa, frequency)
            values.append(value)
        out[name+'_frequency_hz'] = f
        out[name+'_gamma'] = np.array(values)
    np.savez_compressed(ROOT/'data/propagation_measured.npz', **out)
    return out


def measured(rebuild=False):
    verify_sources()
    path = ROOT/'data/propagation_measured.npz'
    if rebuild or not path.exists():
        return extract_measurements()
    with np.load(path) as a:
        return {key: a[key] for key in a.files}
