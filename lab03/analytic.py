"""New seeded realizations of the published CNO/RPB exact wave family.

Not the archived CNO test cases: coefficients are generated here and saved.
Formula: Raonic et al., arXiv:2302.01178v2, Appendix C.3.2.
"""
import numpy as np
from .model import ROOT


def coefficients(seed, modes=24, decay=1.):
    rng=np.random.default_rng(seed)
    i=np.arange(1,modes+1,dtype=float)
    w=i[:,None]**2+i[None,:]**2
    return np.pi/modes**2*rng.uniform(-1,1,(modes,modes))*w**(-decay)


def evaluate(A,x,t,c=.1):
    """Pressure and exact derivatives; velocity initial condition is p_t=0."""
    i=np.arange(1,len(A)+1,dtype=float); k=np.pi*i
    S=np.sin(x[:,None]*k); C=np.cos(x[:,None]*k)
    omega=c*np.sqrt(k[:,None]**2+k[None,:]**2)
    out=[]
    for name in ('p','pt','ptt','xx','yy','xy'):
        values=[]
        for ti in t:
            B=A*np.cos(omega*ti)
            left=right=S
            if name=='pt': B=-A*omega*np.sin(omega*ti)
            elif name=='ptt': B=-omega**2*B
            elif name=='xx': B=-k[:,None]**2*B
            elif name=='yy': B=-k[None,:]**2*B
            elif name=='xy': B=k[:,None]*k[None,:]*B;left=right=C
            values.append(left@B@right.T)
        out.append(np.asarray(values))
    return dict(zip(('p','pt','ptt','xx','yy','xy'),out))


def feature_set(seed,modes=24,decay=1.,noise=0.,A=None,pressure_scale=None):
    A=coefficients(seed,modes,decay) if A is None else A
    x=np.linspace(0,1,34)[1:-1]
    t=np.linspace(.1,4.9,45)
    z=evaluate(A,x,t)
    P=float(np.max(np.abs(evaluate(A,x,[0.])['p']))) if pressure_scale is None else float(pressure_scale)
    X=np.stack((.01*z['xx'],.01*z['yy'],.01*z['xy'],
                -z['pt'],-z['p'],-z['p']**3/P**2),axis=-1)
    y=z['ptt']
    if noise:
        # Observation perturbation on the moment equation, not new physics.
        rng=np.random.default_rng(seed+90000)
        residual=rng.normal(size=y.shape)
        from scipy.ndimage import gaussian_filter1d
        residual=gaussian_filter1d(residual,1.5,axis=0)
        residual*=noise*np.sqrt(np.mean(y*y))/np.sqrt(np.mean(residual*residual))
        y=y+residual
    shape=y.shape
    group=np.broadcast_to(np.arange(shape[0])[:,None,None],shape)
    return {'X':X.reshape(-1,6),'y':y.ravel(),'group':group.ravel(),
        'shape':shape,'source':f'CNO_formula_seed_{seed}_K{modes}_r{decay}',
            'A':A,'seed':seed,'modes':modes,'decay':decay,'pressure_scale':P,'dt_group':float(t[1]-t[0])}


def published_feature_set(case,noise=0.,pressure_scale=None):
    with np.load(ROOT/'data/cno_published_cases.npz') as d:A=d[f'case_{case}_A']
    result=feature_set(4100+case,modes=32,decay=.85,noise=noise,A=A,pressure_scale=pressure_scale)
    result['source']=f'CNO_published_OOD_case_{case}'
    return result


def save_coefficients():
    seeds=[3101,3102,3103,3104,3105,3106,55001,55002]
    np.savez_compressed(ROOT/'data/analytic_coefficients.npz',
        **{f'seed_{s}':coefficients(s,32 if s in (3105,55001,55002) else 24,.85 if s in (3105,55001,55002) else 1.) for s in seeds})


def archive_audit():
    """Verify three actual published OOD cases against the analytical formula.

    The actual OOD archive stores original amplitudes despite also providing
    min/max metadata. Recover sine coefficients by DST-I on 62 interior nodes.
    Only 32x32 active modes are retained, as specified in the archive.
    """
    import h5py
    from scipy.fft import dstn
    path=ROOT.parent/'tmp/lab03/WaveData_64x64_OUT.h5'
    if not path.exists():
        compact=ROOT/'data/cno_published_cases.npz'
        if not compact.exists():return []
        from .model import relative_error
        with np.load(compact) as d:
            return [{'case':i,'endpoint_relative_error':relative_error(d[f'case_{i}_output'],
                evaluate(d[f'case_{i}_A'],np.linspace(0,1,64),[5.])['p'][0])} for i in (0,1,2)]
    records=[];saved={}
    x=np.linspace(0,1,64)
    with h5py.File(path,'r') as f:
        for i in (0,1,2):
            group=f[f'Sample_{i}_t_5']
            initial=group['input'][()]
            target=group['output'][()]
            A=dstn(initial[1:-1,1:-1],type=1)/(63**2)
            A=A[:32,:32]
            calculated=evaluate(A,x,[5.])['p'][0]
            from .model import relative_error
            records.append({'case':i,'endpoint_relative_error':relative_error(target,calculated)})
            saved[f'case_{i}_A']=A;saved[f'case_{i}_input']=initial;saved[f'case_{i}_output']=target
    np.savez_compressed(ROOT/'data/cno_published_cases.npz',**saved)
    import hashlib,json
    compact=ROOT/'data/cno_published_cases.npz'
    compact.with_suffix('.json').write_text(json.dumps({
        'archive':'https://zenodo.org/records/10406879/files/data.zip',
        'member':'data/WaveData_64x64_OUT.h5','cases':[0,1,2],
        'member_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'compact_sha256':hashlib.sha256(compact.read_bytes()).hexdigest(),
        'normalization':'Original stored amplitudes; no additional min-max inverse transform.',
        'status':'Published analytic PDE data, not physical measurements.'},indent=2))
    return records
