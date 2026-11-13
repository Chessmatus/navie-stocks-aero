"""Run using the isolated Clawpack interpreter after comparison selection."""
from pathlib import Path
import json,hashlib
import numpy as np
from clawpack_runner import solve_arrays,SOLVER_VERSION
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/criterion_comparison'


def main():
    plan=json.loads((OUT/'selection.json').read_text())['well_forward_plan']
    sample=plan['test_sample'];scale=plan['pressure_scale']
    path=ROOT/'data'/f'well_test_{sample}.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as d:p=d['p'];rho=d['rho'];c=d['c'];times=d['t'].astype(float)[::2]
    for mask,theta in plan['models'].items():
        out=OUT/f'well_mask{mask}_test{sample}.npz'
        current=False
        if out.exists():
            with np.load(out) as d:
                current=(np.array_equal(d['theta'],theta) and str(d['input_sha256'])==digest
                    and str(d['solver_version'])==SOLVER_VERSION and float(d['pressure_scale'])==scale)
        if current:continue
        pred=solve_arrays(p,rho,c,times,theta,pressure_scale=scale)
        np.savez_compressed(out,p=pred,t=times,theta=theta,pressure_scale=scale,
            input_sha256=digest,solver_version=SOLVER_VERSION)
        print('comparison test mask',mask,'done',flush=True)


if __name__=='__main__':main()
