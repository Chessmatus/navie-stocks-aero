"""Independent scalar acoustic solver: centered FV flux + Verlet splitting.

Second order in space/time in smooth homogeneous media. Mur open boundary at
upper x/y, zero normal flux at lower x/y, per the actual Well generation code.
This is not the upstream Clawpack MC solver; discrepancies are reported.
"""
import numpy as np
from .model import flux_operators,relative_error
from .analytic import coefficients,evaluate


def solve(p0,rho,bulk,times,theta,dx,boundary='maze',cfl=.45,pressure_scale=None):
    theta=np.asarray(theta,float)
    if theta[0]<0 or theta[1]<0 or theta[2]**2>4*theta[0]*theta[1]+1e-12:
        raise ValueError('Non-positive spatial acoustic tensor')
    p=np.asarray(p0,dtype=float).copy(); w=np.zeros_like(p)
    rho=np.broadcast_to(rho,p.shape).astype(float)
    bulk=np.broadcast_to(bulk,p.shape).astype(float)
    mobility=1/rho
    ax=2/(rho[:-1]+rho[1:]); ay=2/(rho[:,:-1]+rho[:,1:])
    scale=max(float(np.max(np.abs(p0))) if pressure_scale is None else pressure_scale,1e-15)
    cmax=float(np.sqrt(np.max(bulk/rho)*max(theta[0]+theta[1]+abs(theta[2]),1e-6)))
    dtmax=cfl*dx/cmax
    cx=np.sqrt(bulk/rho*theta[0]);cy=np.sqrt(bulk/rho*theta[1])

    def force(q):
        fx=ax*(q[1:]-q[:-1])/dx;fy=ay*(q[:,1:]-q[:,:-1])/dx
        out=np.zeros_like(q)
        out[:-1]+=theta[0]*fx/dx;out[1:]-=theta[0]*fx/dx
        out[:,:-1]+=theta[1]*fy/dx;out[:,1:]-=theta[1]*fy/dx
        out*=bulk
        if theta[2]:
            px=np.gradient(q,dx,axis=0);py=np.gradient(q,dx,axis=1)
            out+=.5*theta[2]*bulk*(np.gradient(mobility*py,dx,axis=0)+np.gradient(mobility*px,dx,axis=1))
        return out-theta[4]*q-theta[5]*q**3/scale**2

    acceleration=force(p)
    frames=[p.astype(np.float32)]
    now=float(times[0]); steps=0
    for target in times[1:]:
        count=max(1,int(np.ceil((target-now)/dtmax)))
        dt=(target-now)/count
        decay=np.exp(-theta[3]*dt/2)
        for _ in range(count):
            w*=decay;w+=.5*dt*acceleration
            previous=p.copy();p+=dt*w
            if boundary=='dirichlet':
                p[[0,-1],:]=0;p[:,[0,-1]]=0
            elif boundary=='maze':
                # First-order characteristic radiation at the open edges.
                rx=(cx[-1]*dt-dx)/(cx[-1]*dt+dx)
                ry=(cy[:,-1]*dt-dx)/(cy[:,-1]*dt+dx)
                p[-1]=previous[-2]+rx*(p[-2]-previous[-1])
                p[:,-1]=previous[:,-2]+ry*(p[:,-2]-previous[:,-1])
            elif boundary!='reflecting': raise ValueError(boundary)
            acceleration=force(p)
            w+=.5*dt*acceleration;w*=decay
            if boundary=='dirichlet':w[[0,-1],:]=0;w[:,[0,-1]]=0
            steps+=1
            if not np.isfinite(p).all() or np.max(np.abs(p))>1e8*scale:
                raise FloatingPointError('Unstable candidate rollout')
        now=float(target);frames.append(p.astype(np.float32))
    return np.asarray(frames),steps


def verification():
    errors=[]
    for n in (33,65,129):
        x=np.linspace(0,1,n);X,Y=np.meshgrid(x,x,indexing='ij')
        p0=np.sin(2*np.pi*X)*np.sin(3*np.pi*Y)
        times=np.linspace(0,2,21);theta=np.array([1.,1.,0.,0.,0.,0.])
        pred,steps=solve(p0,1.,.01,times,theta,1/(n-1),'dirichlet')
        ref=np.cos(.1*np.pi*np.sqrt(13)*times)[:,None,None]*p0
        errors.append({'n':n,'relative_L2':relative_error(ref,pred),'steps':steps})
    ratio=[errors[i]['relative_L2']/errors[i+1]['relative_L2'] for i in (0,1)]
    const,_=solve(np.ones((16,16)),3.,4.,[0,.1],[1,1,0,0,0,0],.1,'reflecting')
    return {'standing_wave':errors,'refinement_ratios':ratio,
        'constant_field_error':float(np.max(abs(const-1))),
        'method':'Centered finite-volume flux, velocity Verlet, Strang damping',
        'limitation':'Different numerical diffusion and open boundary approximation from Clawpack.'}
