"""3D incompressible Navier-Stokes on a staggered MAC grid.

First-order upwind advection, centered diffusion; SSP-RK3 projected stages. Normal
boundary velocities are prescribed, tangential values enter through Dirichlet
velocity ghosts. Pressure is the Lagrange multiplier of the MAC projection;
its Neumann Poisson system is solved with DCT and zero mean.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import dctn, idctn


@dataclass
class MACGrid:
    bounds: tuple
    shape: tuple

    def __post_init__(self):
        self.h = tuple((b-a)/n for (a,b),n in zip(self.bounds,self.shape))
        self.centers = tuple(a+(np.arange(n)+.5)*h for (a,b),n,h in zip(self.bounds,self.shape,self.h))
        self.edges = tuple(np.linspace(a,b,n+1) for (a,b),n in zip(self.bounds,self.shape))
        eigen = [(-4*np.sin(np.pi*np.arange(n)/(2*n))**2/h**2).reshape(tuple(n if j==i else 1 for j in range(3)))
                 for i,(n,h) in enumerate(zip(self.shape,self.h))]
        self.eigen = sum(eigen)
        self.eigen[0,0,0] = 1.0

    def coordinates(self, component=None, boundary=None):
        axes = list(self.centers)
        if component is not None:
            axes[component] = self.edges[component]
        if boundary is not None:
            axis,side = boundary
            axes[axis] = np.array([self.bounds[axis][side]])
        return np.meshgrid(*axes,indexing='ij')

    def face_shape(self, component):
        return tuple(n+(i==component) for i,n in enumerate(self.shape))

    def divergence(self, velocity):
        return sum(np.diff(a,axis=i)/h for i,(a,h) in enumerate(zip(velocity,self.h)))

    def pressure_gradient(self, pressure):
        result=[]
        for axis,h in enumerate(self.h):
            a=np.zeros(self.face_shape(axis));sl=[slice(None)]*3;sl[axis]=slice(1,-1)
            a[tuple(sl)]=np.diff(pressure,axis=axis)/h
            result.append(a)
        return result

    def poisson(self, rhs):
        rhs=rhs-rhs.mean()
        transformed=dctn(rhs,type=2,norm='ortho')/self.eigen
        transformed[0,0,0]=0.0
        return idctn(transformed,type=2,norm='ortho')

    def centers_velocity(self, velocity):
        result=[]
        for axis,a in enumerate(velocity):
            low=[slice(None)]*3;high=low.copy();low[axis]=slice(None,-1);high[axis]=slice(1,None)
            result.append(.5*(a[tuple(low)]+a[tuple(high)]))
        return np.stack(result,axis=-1)


def corrected_boundary(grid, faces):
    """Minimum uniform outward-normal shift making the sampled net flux zero."""
    result=[a.copy() for a in faces]
    flux=0.;total_area=0.
    for axis,a in enumerate(faces):
        area=np.prod([h for i,h in enumerate(grid.h) if i!=axis])
        lo=np.take(a,0,axis=axis);hi=np.take(a,-1,axis=axis)
        flux+=(hi.sum()-lo.sum())*area
        total_area+=2*lo.size*area
    delta=flux/total_area
    for axis,a in enumerate(result):
        lo=[slice(None)]*3;hi=lo.copy();lo[axis]=0;hi[axis]=-1
        a[tuple(lo)]+=delta;a[tuple(hi)]-=delta
    return result,float(delta)


def impose_normal(velocity, boundary):
    for axis,a in enumerate(velocity):
        for index in (0,-1):
            sl=[slice(None)]*3;sl[axis]=index
            a[tuple(sl)]=boundary[axis][tuple(sl)]


def project(grid, velocity, boundary, effective_dt=1.):
    velocity=[a.copy() for a in velocity]
    impose_normal(velocity,boundary)
    pressure=grid.poisson(grid.divergence(velocity)/effective_dt)
    gradient=grid.pressure_gradient(pressure)
    velocity=[a-effective_dt*b for a,b in zip(velocity,gradient)]
    return velocity,pressure


def derivative(grid, array, component, axis, ghosts, advector=None):
    """Centered gradient and Laplacian contribution, with physical BC ghosts."""
    a=np.moveaxis(array,axis,0)
    if axis==component:
        left=2*a[0]-a[1];right=2*a[-1]-a[-2]
    else:
        left=2*ghosts[(component,axis,0)]-a[0]
        right=2*ghosts[(component,axis,1)]-a[-1]
    padded=np.concatenate([left[None],a,right[None]],axis=0)
    h=grid.h[axis]
    first=(padded[2:]-padded[:-2])/(2*h)
    if advector is not None:
        speed=np.moveaxis(advector,axis,0)
        first=np.where(speed>=0,(padded[1:-1]-padded[:-2])/h,
                       (padded[2:]-padded[1:-1])/h)
    second=(padded[2:]-2*padded[1:-1]+padded[:-2])/h**2
    return np.moveaxis(first,0,axis),np.moveaxis(second,0,axis)


def momentum_rhs(grid, velocity, ghosts, nu):
    center=grid.centers_velocity(velocity)
    result=[]
    for component,a in enumerate(velocity):
        advectors=[]
        for direction in range(3):
            if direction==component:
                advectors.append(a)
            else:
                c=np.moveaxis(center[...,direction],component,0)
                b=np.concatenate([c[:1],.5*(c[1:]+c[:-1]),c[-1:]],axis=0)
                advectors.append(np.moveaxis(b,0,component))
        rhs=np.zeros_like(a)
        for axis in range(3):
            first,second=derivative(grid,a,component,axis,ghosts,advectors[axis])
            rhs+=-advectors[axis]*first+nu*second
        # Boundary-normal values are prescribed, not integrated by this operator.
        sl=[slice(None)]*3;sl[component]=0;rhs[tuple(sl)]=0
        sl[component]=-1;rhs[tuple(sl)]=0
        result.append(rhs)
    return result


def solve_window(grid, provider, times, nu, cfl=.35, max_dt=None):
    """Integrate a measured window, returning cell velocities and diagnostics.

    provider(t) -> (full sampled MAC faces, tangential Dirichlet ghost targets).
    Its time interpolation must cover the complete requested window.
    """
    times=np.asarray(times,dtype=float)
    if nu<=0 or not np.isfinite(nu) or times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('Positive viscosity and increasing times are required')
    def boundary(t):
        faces,ghosts=provider(t)
        faces,delta=corrected_boundary(grid,faces)
        return faces,ghosts,delta
    faces,ghosts,delta=boundary(times[0])
    state,pressure=project(grid,faces,faces)
    history=[grid.centers_velocity(state)]
    divs=[float(np.sqrt(np.mean(grid.divergence(state)**2)))]
    shifts=[delta];steps=0;t=float(times[0])
    for target in times[1:]:
        while t < target-1e-13:
            speed=sum(float(np.max(np.abs(a)))/h for a,h in zip(state,grid.h))
            dt=min(target-t,cfl/max(speed,1e-12),.20/(nu*sum(1/h**2 for h in grid.h)))
            if max_dt is not None:dt=min(dt,max_dt)
            if steps>200000:raise RuntimeError('Step budget exhausted')
            b0,g0,_=boundary(t)
            k0=momentum_rhs(grid,state,g0,nu)
            b1,g1,_=boundary(t+dt)
            u1,_=project(grid,[u+dt*k for u,k in zip(state,k0)],b1,dt)
            k1=momentum_rhs(grid,u1,g1,nu)
            bh,gh,_=boundary(t+dt/2)
            u2,_=project(grid,[.75*u+.25*(a+dt*k) for u,a,k in zip(state,u1,k1)],bh,.25*dt)
            k2=momentum_rhs(grid,u2,gh,nu)
            state,pressure=project(grid,[(u+2*(a+dt*k))/3 for u,a,k in zip(state,u2,k2)],b1,2*dt/3)
            if not all(np.isfinite(a).all() for a in state):raise FloatingPointError('Nonfinite velocity')
            t+=dt;steps+=1
        history.append(grid.centers_velocity(state))
        divs.append(float(np.sqrt(np.mean(grid.divergence(state)**2))))
        shifts.append(boundary(target)[2])
    return {'velocity':np.stack(history),'pressure_multiplier_final':pressure,'steps':steps,
            'divergence_rms':np.array(divs),'boundary_normal_shift':np.array(shifts)}
