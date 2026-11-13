"""Sine-Galerkin + DOP853 for the homogeneous Dirichlet wave benchmark.

All six candidate terms use the same basis/integrator. Pressure L2 equals
coefficient Frobenius norm by orthogonality. Cubic terms use dealiased sine
quadrature. The original wave is checked against the published exact formula.
"""
import numpy as np
from scipy.integrate import solve_ivp


def first_derivative_matrix(modes):
    i=np.arange(1,modes+1)[:,None];j=np.arange(1,modes+1)[None,:]
    a=i+j;b=i-j
    return j*((1.-(-1.)**a)/a+np.divide(1.-(-1.)**b,b,out=np.zeros_like(b,dtype=float),where=b!=0))


def solve_modes(A,times,theta,pressure_scale=None):
    A=np.asarray(A,float);theta=np.asarray(theta,float)
    m=len(A);k=np.pi*np.arange(1,m+1)
    D=first_derivative_matrix(m)
    nq=2*m+1;x=np.arange(1,nq+1)/(nq+1)
    S=np.sin(x[:,None]*k)
    scale=max(float(np.max(abs(S@A@S.T))) if pressure_scale is None else pressure_scale,1e-15)
    frequency=.01*(theta[0]*k[:,None]**2+theta[1]*k[None,:]**2)+theta[4]
    def rhs(t,state):
        q=state[:m*m].reshape(m,m);v=state[m*m:].reshape(m,m)
        acc=-frequency*q-theta[3]*v
        if abs(theta[2])>1e-12:acc+=.01*theta[2]*(D@q@D.T)
        if abs(theta[5])>1e-12:
            field=S@q@S.T
            acc-=theta[5]*4/(nq+1)**2*(S.T@(field**3)@S)/scale**2
        if not np.isfinite(acc).all() or np.linalg.norm(q)>1e4*max(np.linalg.norm(A),1e-15):
            raise FloatingPointError('Modal candidate grows beyond validation bound')
        return np.r_[v.ravel(),acc.ravel()]
    solution=solve_ivp(rhs,(float(times[0]),float(times[-1])),np.r_[A.ravel(),np.zeros(m*m)],
        method='DOP853',t_eval=times,rtol=1e-9,atol=scale*1e-11)
    if not solution.success:raise RuntimeError(solution.message)
    return solution.y[:m*m].T.reshape(-1,m,m)


def exact_modes(A,times):
    k=np.pi*np.arange(1,len(A)+1)
    omega=.1*np.sqrt(k[:,None]**2+k[None,:]**2)
    return np.cos(np.asarray(times)[:,None,None]*omega)*A


def relative_modal_error(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def validation_catalog(records,A,extra_arrays,pressure_scale):
    """Separate initial fields, each with its own exact reference trajectory."""
    times=np.linspace(0,5,51);reference=exact_modes(A,times)
    extra_references=[exact_modes(extra,times) for extra in extra_arrays]
    answer={}
    for row in records:
        if not row['physical_tensor']:continue
        theta=row['parameters']
        try:
            pred=solve_modes(A,times,theta,pressure_scale=pressure_scale)
            base=relative_modal_error(pred,reference)
            if base>.5:raise ValueError('Validation pressure error exceeds 0.5')
            errors=[relative_modal_error(solve_modes(extra,times,theta,pressure_scale=pressure_scale),ref)
                for extra,ref in zip(extra_arrays,extra_references)]
            answer[str(row['mask'])]={'success':True,'pressure_error':base,
                'intervention_error':float(np.mean(errors)),'transfer_errors':errors,
                'parameters':list(theta),'pressure_scale':pressure_scale,
                'solver':'Sine-Galerkin, DOP853, rtol=1e-9'}
        except (ValueError,FloatingPointError,RuntimeError,OverflowError) as ex:
            answer[str(row['mask'])]={'success':False,'failure':str(ex)}
    return answer
