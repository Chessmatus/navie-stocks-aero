"""Exact rejection sampling on a finite graph family, with staged thresholds.

Within a round: G~Uniform(0,...,63), U~Uniform(0,1), accept iff the
mathematical gate holds and U <= exp(-J(G)/T). Since J>=0, envelope M=1.
Accepted samples have pmf proportional to gate(G)*exp(-J/T).
Each round freezes T and all scores. No pooled stationary-posterior claim.
"""
import numpy as np
import json
from .model import ROOT,fit,bits


def summary(data):
    X,y=data['X'],data['y']
    return X.T@X, X.T@y, float(y@y)


def error(stats,p):
    xx,xy,yy=stats
    return float(np.sqrt(max(0.,(p@xx@p-2*p@xy+yy)/max(yy,1e-30))))


def transformed(data,amplitude=1.,swap=False):
    X=data['X'].copy()*amplitude
    X[:,5]*=amplitude**2
    if swap:X[:,[0,1]]=X[:,[1,0]]
    return {**data,'X':X,'y':data['y']*amplitude}


def catalog(train,valid,interventions,rule,use_interventions=True,forward=None):
    xx,xy,_=summary(train)
    from .statistics import moment_fit
    valstats=summary(valid); intervention_stats=[summary(z) for z in interventions]
    removable=rule['removable'];records=[]
    config=json.loads((ROOT/'criterion.json').read_text())
    for mask in range(64):
        p=moment_fit(xx,xy,mask)
        e=error(valstats,p);ei=float(np.mean([error(s,p) for s in intervention_stats]))
        complexity=int(bits(mask).sum())
        valid_tensor=p[0]>=0 and p[1]>=0 and p[2]**2<=4*p[0]*p[1]+1e-12
        gate=valid_tensor
        failures=[]
        if not valid_tensor:failures.append('nonpositive_spatial_tensor')
        risk=sum(not removable[j] for j in range(6) if not bits(mask)[j])/6
        f=forward.get(str(mask)) if forward is not None else None
        if f is not None and not f.get('success',False):
            gate=False;failures.append('forward_solver_failure')
        pressure=f['pressure_error'] if f is not None and f.get('success') else None
        if pressure is not None and pressure>config['max_validation_pressure_error']:
            gate=False;failures.append('validation_pressure_error_above_limit')
        eint=f['intervention_error'] if f is not None and f.get('success') else ei
        J=(config['moment_weight']*e+config['complexity_weight_per_branch']*complexity+
            config['bootstrap_weight']*risk+(0. if pressure is None else config['pressure_weight']*pressure)
            +(config['intervention_weight']*eint if use_interventions else 0.))
        records.append({'mask':mask,'parameters':p.tolist(),'validation':e,
            'intervention':ei,'complexity':complexity,'J':J,'gate':bool(gate),
            'physical_tensor':bool(valid_tensor),'gate_failures':failures,
            'bootstrap_risk':float(risk),'validation_pressure':pressure,
            'intervention_pressure':None if f is None else f.get('intervention_error'),
            'forward_verified':bool(f is not None and f.get('success'))})
    return records


def sample_round(records,T,n=5000,seed=20261112):
    if T<=0:raise ValueError('Temperature must be positive')
    rng=np.random.default_rng(seed)
    proposal=rng.integers(0,len(records),size=n)
    uniform=rng.random(n)
    scores=np.array([r['J'] for r in records]);gate=np.array([r['gate'] for r in records])
    prob=gate*np.exp(-scores/T)
    accept=uniform<=prob[proposal]
    reason=np.where(~gate[proposal],1,np.where(accept,0,2)).astype(np.int8)
    expected=prob/prob.sum()
    counts=np.bincount(proposal[accept],minlength=len(records))
    empirical=counts/max(int(accept.sum()),1)
    return {'proposal':proposal,'uniform':uniform,'accepted':accept,'reason':reason,'T':T,'seed':seed,
        'expected_pmf':expected,'empirical_pmf':empirical,'acceptance_rate':float(accept.mean()),
        'expected_acceptance_rate':float(prob.mean()),
        'total_variation':float(.5*np.abs(expected-empirical).sum())}


def run_sampler(records,seed=20261112):
    if any(r['gate'] and not r.get('forward_verified',True) for r in records):
        raise ValueError('Forward validation must be complete before graph selection')
    # Pilot score scale; subsequent temperatures use the preceding accepted scores.
    T=float(np.median([r['J'] for r in records if r['gate']]))
    rounds=[]
    for k in range(3):
        r=sample_round(records,T,seed=seed+k)
        rounds.append(r)
        accepted_scores=np.array([records[i]['J'] for i in r['proposal'][r['accepted']]])
        if len(accepted_scores):T=max(.04,min(T*.65,float(np.quantile(accepted_scores,.25))))
    accepted=np.concatenate([r['proposal'][r['accepted']] for r in rounds])
    if not len(accepted):raise RuntimeError('No accepted graph; increase the fixed attempt budget')
    selected=min(set(accepted),key=lambda i:(records[i]['J'],records[i]['complexity'],i))
    return rounds,int(selected)
