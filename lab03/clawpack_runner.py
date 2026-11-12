"""Serial Clawpack forward solves, including all six candidate branches.

Diagonal tensor via exact coordinate scaling; mixed tensor contribution and
reaction memory via Strang-split RK4 source. For m=p_t+K div(u),
  p_t=-K div(u)+m, u_t=-A grad(p)/rho-d*u,
  m_t=-d*m-r*p-n*p**3/P0**2
implies exactly p_tt=K div(A grad(p)/rho)-d*p_t-r*p-n*p**3/P0**2.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/lab03/mpl_claw'))
import json
import argparse
import hashlib
import numpy as np

SOLVER_VERSION='clawpack-5.9-mc-ssprk-source-v2.1'
ROOT=Path(__file__).resolve().parent


def serial_backend():
    import mpi4py,sys
    mpi4py.rc.initialize=False;mpi4py.rc.finalize=False
    # Optional PETSc import in PyClaw's run_serialized decorator would start
    # MPI even for a serial calculation. This process exclusively uses PyClaw.
    sys.modules['petsc4py']=None
    from clawpack import pyclaw,riemann
    import logging
    logging.disable(logging.INFO)
    for name in logging.root.manager.loggerDict:
        if name.startswith(('CLAW','clawpack')):logging.getLogger(name).setLevel(logging.WARNING)
    return pyclaw,riemann


def solve_arrays(pressure,rho,c,times,theta,amplitude=1.,swap=False,max_error=None):
    pyclaw,riemann=serial_backend()
    theta=np.asarray(theta,float)
    if theta[0]<-1e-10 or theta[1]<-1e-10 or theta[2]**2>4*theta[0]*theta[1]+1e-12:
        raise ValueError('Nonpositive spatial tensor')
    if swap:pressure=pressure.transpose(0,2,1);rho=rho.T;c=c.T
    initial=amplitude*pressure[0]
    scale=max(float(np.max(abs(pressure[0]))),1e-15)  # fixed before intervention
    sx,sy=np.sqrt(np.maximum(theta[:2],1e-12))
    solver=pyclaw.ClawSolver2D(riemann.vc_acoustics_2D)
    solver.dimensional_split=False;solver.limiters=pyclaw.limiters.tvd.MC
    solver.cfl_desired=.9;solver.cfl_max=1.
    for j in (0,1):
        solver.bc_lower[j]=pyclaw.BC.wall;solver.bc_upper[j]=pyclaw.BC.extrap
        solver.aux_bc_lower[j]=pyclaw.BC.wall;solver.aux_bc_upper[j]=pyclaw.BC.extrap
    n=rho.shape[0];dx=2/n
    if abs(theta[3])>1e-12 and np.max(abs(theta[[2,4,5]]))<=1e-12:
        def damping(solver,state,dt):state.q[1:]*=np.exp(-theta[3]*dt)
        solver.step_source=damping;solver.source_split=2
    elif np.max(abs(theta[2:]))>1e-12:
        def source(solver,state,dt):
            q=state.q;m=state.problem_data['memory']
            def rhs(q,m):
                dq=np.empty_like(q);dq[0]=m
                dq[1]=-theta[3]*q[1]-theta[2]/(2*sx*rho)*np.gradient(q[0],dx,axis=1)
                dq[2]=-theta[3]*q[2]-theta[2]/(2*sy*rho)*np.gradient(q[0],dx,axis=0)
                dm=-theta[3]*m-theta[4]*q[0]-theta[5]*q[0]**3/scale**2
                return dq,dm
            a,b=rhs(q,m);aa,bb=rhs(q+dt*a/2,m+dt*b/2)
            aaa,bbb=rhs(q+dt*aa/2,m+dt*bb/2)
            aaaa,bbbb=rhs(q+dt*aaa,m+dt*bbb)
            state.q[:]=q+dt/6*(a+2*aa+2*aaa+aaaa)
            state.problem_data['memory']=m+dt/6*(b+2*bb+2*bbb+bbbb)
            if not np.isfinite(state.q).all() or np.max(abs(state.q[0]))>50*scale:
                raise FloatingPointError('Candidate source grows beyond the validation bound')
        solver.step_source=source;solver.source_split=2
        rate=abs(theta[3])+np.sqrt(abs(theta[4])+48*abs(theta[5]))
        solver.dt_max=min(.04,1./max(rate,1e-12))
    domain=pyclaw.Domain([pyclaw.Dimension(-1/s,1/s,n,name=axis) for s,axis in [(sx,'x'),(sy,'y')]])
    state=pyclaw.State(domain,3,2)
    state.aux[0]=rho;state.aux[1]=c;state.q[0]=initial;state.q[1:]=0
    state.problem_data['memory']=np.zeros_like(rho)
    if max_error is not None:
        reference=amplitude*pressure[::2]
        denominator=float(np.sum(reference.astype(float)**2))
        seen=set();accumulated=[0.]
        def early_bound(solver,state):
            index=int(np.argmin(abs(times-state.t)))
            if index not in seen and abs(times[index]-state.t)<1e-9:
                seen.add(index)
                accumulated[0]+=float(np.sum((state.q[0]-reference[index])**2))
                # Future squared errors are nonnegative: this is a certified
                # lower bound on the complete validation trajectory error.
                if accumulated[0]>max_error**2*denominator:
                    raise ValueError(f'Validation pressure error necessarily exceeds {max_error}')
        solver.before_step=early_bound
    controller=pyclaw.Controller();controller.solution=pyclaw.Solution(state,domain)
    controller.solver=solver;controller.keep_copy=True;controller.output_format=None
    controller.tfinal=float(times[-1]);controller.output_style=2;controller.out_times=times.tolist()
    controller.run()
    prediction=np.array([f.q[0] for f in controller.frames],dtype=np.float32)
    actual=np.array([f.t for f in controller.frames])
    if len(actual)!=len(times) or not np.allclose(actual,times,atol=1e-9):
        raise RuntimeError('Forward solver did not reach the requested times')
    return prediction


def run_case(split,model,sample=1):
    path=ROOT/'data'/f'well_{split}_{sample}.npz'
    with np.load(path) as d:pressure=d['p'];rho=d['rho'];c=d['c'];times=d['t'].astype(float)[::2]
    theta=([1.,1.,0.,0.,0.,0.] if model=='original' else
        json.loads((ROOT/'results/report.json').read_text())['well']['selected']['parameters'])
    prediction=solve_arrays(pressure,rho,c,times,theta)
    reference=pressure[::2]
    out=ROOT/'results'/f'clawpack_{split}{sample}_{model}.npz'
    np.savez_compressed(out,p=prediction,t=times,theta=theta,sample=sample,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),solver_version=SOLVER_VERSION)
    print(json.dumps({'split':split,'sample':sample,'model':model,
        'relative_L2':float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference)),
        'final_relative_L2':float(np.linalg.norm(prediction[-1]-reference[-1])/np.linalg.norm(reference[-1]))}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='test');p.add_argument('--model',default='original')
    p.add_argument('--sample',type=int,default=1)
    a=p.parse_args();run_case(a.split,a.model,a.sample)
