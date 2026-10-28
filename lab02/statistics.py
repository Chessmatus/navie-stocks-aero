"""Paired moving-block pairs bootstrap and conditional residual interventions."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import wasserstein_distance
from .model import fit, predict, NULL


def train_mask(n, block=8):
    # Every fourth contiguous band is held out. No validation values set scales.
    return (np.arange(n)//block)%4 != 3


def block_indices(mask, length, rng, size=None):
    """Consecutive original frequency rows; blocks never bridge holdout gaps."""
    mask = np.asarray(mask, dtype=bool)
    if length < 1 or length > len(mask):
        raise ValueError('Invalid block length')
    starts = np.flatnonzero(np.convolve(mask.astype(int), np.ones(length, dtype=int), mode='valid') == length)
    if not len(starts):
        raise ValueError('No full block in training sample')
    size = int(mask.sum()) if size is None else size
    return np.concatenate([np.arange(j, j+length) for j in rng.choice(starts, int(np.ceil(size/length)))])[:size]


def conditional_scale(f, y, mask):
    """Pilot residual scale conditional on f; train-only, not an instrument spec."""
    p = fit(f[mask], y[mask], [1., 1.])
    residual = y-predict(f, p)
    r = residual[mask]
    global_scale = np.maximum(1.4826*np.median(np.abs(r-np.median(r, axis=0)), axis=0), 1e-7)
    # Rolling power on training frequencies; interpolate only the nuisance scale.
    power = uniform_filter1d(r*r, size=17, axis=0, mode='nearest')
    local = np.maximum(np.sqrt(power), global_scale*.5)
    scale = np.column_stack([np.interp(f, f[mask], local[:, j]) for j in range(2)])
    z = np.zeros_like(y)
    z[mask] = residual[mask]/scale[mask]
    z[mask] -= np.mean(z[mask], axis=0)
    return scale, z


def distances(a, b, normalizer, directions):
    a, b = np.asarray(a), np.asarray(b)
    raw = np.array([wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])])
    az, bz = a/normalizer, b/normalizer
    # Fixed finite projection approximation, not full multidimensional W1.
    sliced = np.mean([wasserstein_distance(az@v, bz@v) for v in directions])
    return raw, raw/normalizer, float(sliced)


def score(f, y, scale, p):
    return float(np.sqrt(np.mean(((predict(f, p)-y)/scale)**2)))


def bootstrap(f, y, mask, scale, z, n_boot=400, block=8, seed=20261021,
              loss='linear', strengths=(0., .5, 1., 2., 4.), variables=('alpha','beta','both')):
    rng = np.random.default_rng(seed)
    n = len(f)
    samples = np.empty((len(variables), len(strengths), n_boot, 3))
    baseline = np.empty((n_boot, 3))
    # Paired interventions: identical bootstrap rows and noise draw across lambda.
    for b in range(n_boot):
        ids = block_indices(mask, block, rng)
        noise_ids = block_indices(mask, block, rng, size=n)
        noise = z[noise_ids]*scale
        baseline[b] = fit(f[ids], y[ids], scale[ids], loss)
        for v, variable in enumerate(variables):
            change = noise.copy()
            if variable == 'alpha': change[:, 1] = 0
            elif variable == 'beta': change[:, 0] = 0
            elif variable != 'both': raise ValueError('Unknown observed variable')
            for l, strength in enumerate(strengths):
                if strength == 0:
                    samples[v, l, b] = baseline[b]
                else:
                    samples[v, l, b] = fit(f[ids], (y+strength*change)[ids], scale[ids], loss)
    return baseline, samples


def ablation(f, y, mask, scale, baseline, block=8, seed=100, n_boot=400,
             tolerance=.25, stability=None):
    """Independent holdout error + paired bootstrap uncertainty for pruning.

    Removal requires coefficient null in 95% percentile interval, upper 95%
    holdout RMS cost <= 0.25 residual units, and intervention robustness <= 1 SD.
    Tolerances are educational choices, never calibrated statistical tests.
    """
    rng = np.random.default_rng(seed)
    val = ~mask
    transitions = np.flatnonzero(np.diff(np.r_[False, val, False]))
    longest_validation_band = int(np.max(transitions[1::2]-transitions[::2]))
    p = fit(f[mask], y[mask], scale[mask])
    full_score = score(f[val], y[val], scale[val], p)
    ci = np.quantile(baseline, [.025,.975], axis=0)
    records = []
    for j in range(3):
        q = fit(f[mask], y[mask], scale[mask], fixed={j: NULL[j]})
        delta = []
        for b in range(n_boot):
            tr = block_indices(mask, block, rng)
            va = block_indices(val, min(block, longest_validation_band), rng)
            p_b = fit(f[tr], y[tr], scale[tr])
            q_b = fit(f[tr], y[tr], scale[tr], fixed={j: NULL[j]})
            delta.append(score(f[va], y[va], scale[va], q_b)-score(f[va], y[va], scale[va], p_b))
        upper = float(np.quantile(delta, .95))
        includes = bool(ci[0,j] <= NULL[j] <= ci[1,j])
        robust = stability is None or stability[j] <= 1.
        remove = includes and upper <= tolerance and robust
        records.append({'parameter': ('n','a','b')[j], 'null': float(NULL[j]),
                        'ci95': ci[:,j].tolist(), 'includes_null': includes,
                        'reduced_parameters': q.tolist(), 'holdout_full': full_score,
                        'holdout_reduced': score(f[val], y[val], scale[val], q),
                        'delta_upper95': upper, 'tolerance': tolerance,
                        'stability_W1_SD': None if stability is None else float(stability[j]),
                        'decision': 'remove' if remove else 'keep'})
    return records
