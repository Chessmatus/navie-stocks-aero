"""Diagnostic only: an exact translated 3D Beltrami solution, no experimental data."""
from pathlib import Path
import json
import numpy as np
from lab01.solver import MACGrid, solve_window
from lab01.experiment import BOUNDS

NU=9e-7;DRIFT=.25;AMPLITUDE=.04;END=.09
rows=[]
for wavelength in (.032,.016,.008):
    k=2*np.pi/wavelength
    def field(coords,t):
        x,y,z=coords
        vortex=AMPLITUDE*np.exp(-NU*k*k*t)*np.stack([
            np.sin(k*z)+np.cos(k*y),
            np.sin(k*(x-DRIFT*t))+np.cos(k*z),
            np.sin(k*y)+np.cos(k*(x-DRIFT*t))],axis=-1)
        return vortex+np.array([DRIFT,0.,0.])
    for shape in ((36,24,24),(72,48,48)):
        grid=MACGrid(BOUNDS,shape)
        def provider(t):
            faces=[field(grid.coordinates(c),t)[...,c] for c in range(3)]
            ghosts={(c,a,s):np.squeeze(field(grid.coordinates(c,(a,s)),t)[...,c],axis=a)
                    for c in range(3) for a in range(3) if a!=c for s in (0,1)}
            return faces,ghosts
        r=solve_window(grid,provider,np.linspace(0,END,10),NU,cfl=.35)
        exact=field(grid.coordinates(),END);pred=r['velocity'][-1]
        # Fixed physical interior ROI, equal exclusion for each grid.
        coords=grid.coordinates();mask=np.ones(grid.shape,dtype=bool)
        for c,(lo,hi) in zip(coords,((.035,.085),(-.015,.015),(-.015,.015))):mask&=(c>=lo)&(c<=hi)
        error=(pred-exact)[mask];truth=exact[mask];fluct=truth-[DRIFT,0,0]
        row={'wavelength_m':wavelength,'grid':shape,'nu':NU,'t_end_s':END,
             'relative_total_velocity_error':float(np.linalg.norm(error)/np.linalg.norm(truth)),
             'relative_fluctuation_error':float(np.linalg.norm(error)/np.linalg.norm(fluct)),
             'component_relative_errors':np.sqrt(np.sum(error**2,axis=0)/np.sum(truth**2,axis=0)).tolist(),
             'divergence_rms_max':float(r['divergence_rms'].max()),'steps':r['steps']}
        rows.append(row);print(json.dumps(row),flush=True)
Path('outputs/turbine_1L/solver_diagnostic.json').write_text(json.dumps({
    'kind':'analytical_solver_verification_not_experimental_data',
    'exact_solution':'u=(.25,0,0)+A*exp(-nu*k^2*t)*(sin(kz)+cos(ky),sin(k(x-.25t))+cos(kz),sin(ky)+cos(k(x-.25t))); A=.04; p/rho=-|u-(.25,0,0)|^2/2',
    'boundary_policy':'Exact analytical velocity at every integrator stage; no measurement interpolation',
    'roi_bounds_m':[[.035,.085],[-.015,.015],[-.015,.015]],
    'norm_policy':'Final-time cell-center L2 over fixed physical ROI; discrete points differ across grids',
    'results':rows},indent=2))
