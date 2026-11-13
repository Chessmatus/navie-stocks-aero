"""Evaluate every mathematically admissible graph on validation only.

Run using the isolated Clawpack Python. A content-addressed JSON checkpoint
makes notebook reruns offline and fast. No test file is accessed here.
"""
from pathlib import Path
import json,hashlib,gc,time
import numpy as np
from clawpack_runner import solve_arrays,SOLVER_VERSION

ROOT=Path(__file__).resolve().parent


def main():
    candidates=json.loads((ROOT/'results/validation_candidates.json').read_text())
    path=ROOT/'data/well_valid_0.npz'
    config=json.loads((ROOT/'criterion.json').read_text());maximum=config['max_validation_pressure_error']
    signature=hashlib.sha256(path.read_bytes()+json.dumps(candidates,sort_keys=True).encode()+SOLVER_VERSION.encode()+str(maximum).encode()).hexdigest()
    out=ROOT/'results/well_forward_validation.json'
    payload=json.loads(out.read_text()) if out.exists() else {}
    if payload.get('signature')!=signature:payload={'signature':signature,'solver_version':SOLVER_VERSION,'input':path.name,'records':{}}
    with np.load(path) as d:pressure=d['p'];rho=d['rho'];c=d['c'];t=d['t'].astype(float)[::2]
    ref=pressure[::2]
    def error(a,b):return float(np.linalg.norm(a.astype(float)-b)/np.linalg.norm(b))
    for record in candidates:
        mask=record['mask'];key=str(mask)
        if not record['physical_tensor'] or key in payload['records']:continue
        theta=np.array(record['parameters']);started=time.monotonic()
        try:
            p=solve_arrays(pressure,rho,c,t,theta,max_error=maximum)
            base=error(p,ref)
            if base>maximum:raise ValueError(f'Validation pressure error exceeds {maximum}')
            if abs(theta[5])<1e-12:amplitude=base
            else:amplitude=error(solve_arrays(pressure,rho,c,t,theta,amplitude=1.5),1.5*ref)
            swap=error(solve_arrays(pressure,rho,c,t,theta,swap=True),ref.transpose(0,2,1))
            value={'success':True,'pressure_error':base,'amplitude_error':amplitude,
                'swap_error':swap,'intervention_error':(amplitude+swap)/2,
                'parameters':theta.tolist(),'seconds':time.monotonic()-started}
            if mask in (3,11):np.savez_compressed(ROOT/'results'/f'well_validation_mask{mask}.npz',p=p,t=t,theta=theta)
        except (FloatingPointError,ValueError,RuntimeError,OverflowError) as ex:
            value={'success':False,'failure':str(ex),'parameters':theta.tolist()}
        payload['records'][key]=value
        temporary=out.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,indent=2));temporary.replace(out)
        print('validation mask',mask,value.get('pressure_error',value.get('failure')),flush=True)
        gc.collect()
    print('Validation cache complete:',len(payload['records']),'graphs',flush=True)


if __name__=='__main__':
    import fcntl
    lockpath=ROOT.parent/'tmp/lab03/forward_validation.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
