"""Analytical check of the PDE integrator; no experimental claims or plots."""
import numpy as np
from .solver import MACGrid, solve_window


def abc_error(n,max_dt):
    """Relative final error for a decaying 3D Beltrami solution at t=.05."""
    nu=.1;grid=MACGrid(((0,2*np.pi),)*3,(n,)*3)
    def field(coords,t):
        x,y,z=coords
        return np.stack([np.sin(z)+np.cos(y),np.sin(x)+np.cos(z),np.sin(y)+np.cos(x)],axis=-1)*np.exp(-nu*t)
    def provider(t):
        faces=[field(grid.coordinates(c),t)[...,c] for c in range(3)]
        ghosts={(c,a,s):np.squeeze(field(grid.coordinates(c,(a,s)),t)[...,c],axis=a)
                for c in range(3) for a in range(3) if a!=c for s in (0,1)}
        return faces,ghosts
    result=solve_window(grid,provider,[0,.05],nu,max_dt=max_dt)
    exact=field(grid.coordinates(),.05)
    error=np.linalg.norm((result['velocity'][-1]-exact).ravel())/np.linalg.norm(exact.ravel())
    return float(error),float(result['divergence_rms'].max())
