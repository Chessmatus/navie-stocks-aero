"""Frame-block bootstrap and the conservative branch rule from lab 2.

The null is zero for every optional branch. Fixed physical inputs rho,K are
never removed. Tolerances are specified in normalized derivative RMS units.
"""
import numpy as np
from scipy.stats import wasserstein_distance
from .model import fit,bits,relative_error,TERMS


def moments(data):
    X,y,group=data['X'],data['y'],data['group']
    n=int(group.max())+1
    return (np.array([X[group==g].T@X[group==g] for g in range(n)]),
            np.array([X[group==g].T@y[group==g] for g in range(n)]))


def moment_fit(xx,xy,mask=63):
    active=bits(mask); p=np.zeros(6)
    if active.any():
        a=xx[np.ix_(active,active)]
        s=np.sqrt(np.maximum(np.diag(a),1e-30))
        p[active]=np.linalg.lstsq(a/s[:,None]/s[None,:],xy[active]/s,rcond=1e-10)[0]/s
    return p


def frame_ids(n,block,rng):
    starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
    return np.concatenate([np.arange(i,i+block) for i in starts])[:n]


def bootstrap(data,mask=63,n_boot=400,block=8,seed=20261112,perturb=False):
    xx,xy=moments(data); n=len(xx)
    rng=np.random.default_rng(seed)
    rng_noise=np.random.default_rng(seed+7919)
    samples=[]
    X,y=data['X'],data['y']
    p=fit(X,y,mask)
    residual=y-X@p
    # Conditional noise keeps the spatial residual pattern of each frame.
    # Independently signed temporal blocks: zero mean in expectation.
    noise_xy=np.zeros_like(xy)
    if perturb:
        shape=data['shape']; residual=residual.reshape(shape)
        residual=residual-residual.mean(axis=0,keepdims=True)
        for g in range(n):
            noise_xy[g]=X[data['group']==g].T@residual[g].ravel()
    for _ in range(n_boot):
        ids=frame_ids(n,block,rng)
        zz=xy[ids].sum(axis=0)
        if perturb:
            signs=np.repeat(rng_noise.choice([-1.,1.],int(np.ceil(n/block))),block)[:n]
            zz=zz+(noise_xy[ids]*signs[:,None]).sum(axis=0)
        samples.append(moment_fit(xx[ids].sum(axis=0),zz,mask))
    return np.asarray(samples)


def pruning(train,valid,blocks=(4,8,12),n_boot=400):
    records=[]; full=fit(train['X'],train['y'])
    scale=float(np.sqrt(np.mean(valid['y']**2)))
    fullres=valid['X']@full-valid['y']
    for block in blocks:
        base=bootstrap(train,n_boot=n_boot,block=block)
        noisy=bootstrap(train,n_boot=n_boot,block=block,perturb=True)
        ci=np.quantile(base,[.025,.975],axis=0)
        sd=np.std(base,axis=0,ddof=1)
        w=np.array([wasserstein_distance(base[:,j],noisy[:,j])/max(sd[j],1e-10) for j in range(6)])
        for j in range(6):
            reduced=fit(train['X'],train['y'],63^(1<<j))
            redres=valid['X']@reduced-valid['y']
            # Paired validation blocks, keep all spatial cells in each frame.
            fr=np.mean(fullres.reshape(valid['shape'])**2,axis=(1,2))
            rr=np.mean(redres.reshape(valid['shape'])**2,axis=(1,2))
            rng=np.random.default_rng(9000+block)
            deltas=[]
            for _ in range(n_boot):
                ids=frame_ids(len(fr),block,rng)
                deltas.append((np.sqrt(rr[ids].mean())-np.sqrt(fr[ids].mean()))/scale)
            includes=bool(ci[0,j]<=1e-9 and ci[1,j]>=-1e-9)
            upper=float(np.quantile(deltas,.95))
            records.append({'term':TERMS[j],'index':j,'block':block,'ci95':ci[:,j].tolist(),
                'includes_zero':includes,'delta_upper95':upper,'W1_SD':float(w[j]),
                'remove':bool(includes and upper<=.01 and w[j]<=1.)})
    removable=[all(r['remove'] for r in records if r['index']==j) for j in range(6)]
    return {'records':records,'removable':removable,'tolerance':.01,'W1_SD_max':1.,
        'blocks':list(blocks),'n_boot':n_boot,'full_parameters':full.tolist()}
