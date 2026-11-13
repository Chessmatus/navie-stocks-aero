"""Evaluate graphs on base validation and separate stored transfer trajectories.

No amplitude scaling or axis-transpose oracle is used in criterion v3.
All trajectories share the pressure normalization fixed on training data.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from clawpack_runner import solve_arrays,SOLVER_VERSION

ROOT=Path(__file__).resolve().parent


def main():
    candidates=json.loads((ROOT/'results/validation_candidates.json').read_text())
    config=json.loads((ROOT/'criterion.json').read_text());maximum=config['max_validation_pressure_error']
    scale=float(candidates[0]['pressure_scale'])
    samples=[config['validation_well_sample']]+config['transfer_well_samples']
    paths=[ROOT/'data'/f'well_valid_{sample}.npz' for sample in samples]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature=hashlib.sha256(json.dumps({'inputs':hashes,'candidates':candidates,
        'solver':SOLVER_VERSION,'protocol':config},sort_keys=True).encode()).hexdigest()
    out=ROOT/'results/well_forward_validation.json'
    payload=json.loads(out.read_text()) if out.exists() else {}
    if payload.get('signature')!=signature:
        payload={'signature':signature,'criterion_version':3,'solver_version':SOLVER_VERSION,
            'inputs':hashes,'base_sample':samples[0],'transfer_samples':samples[1:],
            'pressure_scale':scale,'records':{}}
    arrays=[]
    for path in paths:
        with np.load(path) as d:
            arrays.append({k:d[k] for k in ('p','rho','c','t')})
    # Reuse only verified v2 base errors for linear candidates. For cubic graphs,
    # the new common training scale changes the PDE, so base solves are repeated.
    legacy_path=ROOT/'results/criterion_v2/well_forward_validation.json'
    legacy={}
    if legacy_path.exists():
        old=json.loads(legacy_path.read_text())
        old_candidates=json.loads((legacy_path.parent/'validation_candidates.json').read_text())
        old_signature=hashlib.sha256(paths[0].read_bytes()+json.dumps(old_candidates,sort_keys=True).encode()
            +old['solver_version'].encode()+str(maximum).encode()).hexdigest()
        if old_signature==old.get('signature'):legacy=old['records']
    def error(a,b):return float(np.linalg.norm(a.astype(float)-b)/np.linalg.norm(b))
    for record in candidates:
        mask=record['mask'];key=str(mask)
        if not record['physical_tensor'] or key in payload['records']:continue
        theta=np.array(record['parameters']);started=time.monotonic()
        try:
            d=arrays[0];times=d['t'].astype(float)[::2]
            previous=legacy.get(key,{})
            reuse=(abs(theta[5])<1e-12 and previous.get('parameters') is not None
                and np.allclose(theta,previous['parameters'],atol=1e-10,rtol=1e-10))
            if reuse and not previous.get('success'):raise ValueError(previous['failure'])
            if reuse:base=previous['pressure_error']
            else:
                p=solve_arrays(d['p'],d['rho'],d['c'],times,theta,max_error=maximum,pressure_scale=scale)
                base=error(p,d['p'][::2])
                if mask in (3,11):np.savez_compressed(ROOT/'results'/f'well_validation_mask{mask}.npz',p=p,t=times,theta=theta)
            if base>maximum:raise ValueError(f'Validation pressure error exceeds {maximum}')
            transfer=[]
            for sample,d in zip(samples[1:],arrays[1:]):
                pred=solve_arrays(d['p'],d['rho'],d['c'],d['t'].astype(float)[::2],theta,pressure_scale=scale)
                transfer.append({'sample':sample,'pressure_error':error(pred,d['p'][::2])})
            value={'success':True,'pressure_error':base,'transfer_errors':transfer,
                'intervention_error':float(np.mean([r['pressure_error'] for r in transfer])),
                'parameters':theta.tolist(),'pressure_scale':scale,'base_reused_from_v2':bool(reuse),
                'seconds':time.monotonic()-started}
        except (FloatingPointError,ValueError,RuntimeError,OverflowError) as ex:
            value={'success':False,'failure':str(ex),'parameters':theta.tolist(),'pressure_scale':scale}
        payload['records'][key]=value
        temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(out)
        print('validation mask',mask,'base',value.get('pressure_error'),
            'transfer',value.get('intervention_error',value.get('failure')),flush=True)
    print('Validation cache complete:',len(payload['records']),'graphs',flush=True)


if __name__=='__main__':
    import fcntl
    lockpath=ROOT.parent/'tmp/lab03/forward_validation.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
