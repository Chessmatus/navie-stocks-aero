"""Reproduce the FEniCSx comparison in the isolated .fenicsx Conda environment."""
from pathlib import Path
import argparse,json,time,os,sys
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
# Select the provider used by conda-forge's macOS libfabric build tests.
if sys.platform=='darwin':
    os.environ.setdefault('FI_PROVIDER','tcp')
import numpy as np
from lab01.fenics_solver import IPCS,ExperimentalProvider
from lab01.experiment import BOUNDS,observation_points,prepare_measurements,measured_observations
from lab01.calibration import velocity_metrics,fit_viscosity,velocity_residual

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'turbine_1L'/'fenicsx';OUT.mkdir(parents=True,exist_ok=True)
NU_REF=9e-7

def save(name,result):
    (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


def check():
    from mpi4py import MPI
    velocity=np.array([.25,.01,-.01]);points=np.array([[.06,0,0]])
    solver=IPCS(BOUNDS,(4,4,4),points,dt=.001)
    r=solver.solve(lambda p,t:np.tile(velocity,(len(p),1)),[0,.002],NU_REF)
    error=float(np.max(abs(r['prediction']-velocity)))
    print('constant flow error, divergence:',error,r['divergence_rms'].tolist(),flush=True)
    # Iterative mass solves use an absolute residual tolerance. A 1e-9 m/s
    # velocity tolerance corresponds to about 1e-7 /s on this 1 cm mesh.
    assert error<1e-9 and np.max(r['divergence_rms'])<1e-7
    solver.close()
    report={'constant_flow_max_absolute_error_m_per_s':error,
            'max_divergence_rms_s_inv':float(np.max(r['divergence_rms'])),
            'dolfinx_version':r['dolfinx_version'],'mpi_library':MPI.Get_library_version(),
            'FI_PROVIDER':os.environ.get('FI_PROVIDER')}
    # Let PETSc/DOLFINx destruct their communicators before mpi4py finalizes.
    report['mpi_finalize_policy']='automatic at interpreter shutdown'
    save('environment_check',report)
    print({k:v for k,v in report.items() if k!='mpi_library'},flush=True)


def verify():
    from lab01.solver import MACGrid,solve_window
    from lab01.experiment import observe_solution
    points=observation_points();nu=NU_REF;speed=.25;amplitude=.04;rows=[]
    for wavelength in (.032,.016):
        k=2*np.pi/wavelength
        def field(points,t):
            x,y,z=np.asarray(points).T
            vortex=amplitude*np.exp(-nu*k*k*t)*np.stack([np.sin(k*z)+np.cos(k*y),
                    np.sin(k*(x-speed*t))+np.cos(k*z),np.sin(k*y)+np.cos(k*(x-speed*t))],axis=-1)
            return vortex+np.array([speed,0,0])
        exact=field(points,.09)
        def metrics(pred):
            e=pred-exact;fluct=exact-[speed,0,0]
            return {'velocity':velocity_metrics(pred,exact),'relative_fluctuation_error':float(np.linalg.norm(e)/np.linalg.norm(fluct))}
        grid=MACGrid(BOUNDS,(36,24,24))
        def mac_provider(t):
            def sample(coords):return field(np.stack(coords,axis=-1).reshape(-1,3),t).reshape(coords[0].shape+(3,))
            faces=[sample(grid.coordinates(c))[...,c] for c in range(3)]
            ghosts={(c,a,s):np.squeeze(sample(grid.coordinates(c,(a,s)))[...,c],axis=a)
                    for c in range(3) for a in range(3) if c!=a for s in (0,1)}
            return faces,ghosts
        r=solve_window(grid,mac_provider,np.linspace(0,.09,10),nu)
        row={'solver':'MAC upwind','grid':[36,24,24],'wavelength_m':wavelength,
             **metrics(observe_solution(grid,r['velocity'][-1:],points)[0])};rows.append(row);print(row,flush=True)
        for shape,dt in [((12,8,8),.001),((18,12,12),.001),((18,12,12),.0005)]:
            solver=IPCS(BOUNDS,shape,points,dt=dt)
            r=solver.solve(field,np.linspace(0,.09,10),nu,p0=lambda p,t:-.5*np.sum((field(p,t)-[speed,0,0])**2,axis=1),verbose=True)
            row={'solver':'FEniCSx P2/P1 IPCS','grid':list(shape),'dt_s':dt,'wavelength_m':wavelength,
                 **metrics(r['prediction'][-1]),'divergence_rms_max':float(r['divergence_rms'].max()),
                 'seconds':r['seconds'],'velocity_dofs':r['velocity_dofs'],'pressure_dofs':r['pressure_dofs']}
            rows.append(row);print(row,flush=True);solver.close()
            save('verification',{'kind':'exact translated Beltrami solution, not experimental data','nu':nu,'speed':speed,
                 'amplitude':amplitude,'end_time_s':.09,'comparison':'same 539 fixed physical interior points',
                 'fenics_initial_pressure':'exact analytical pressure','results':rows})
    return rows


def experiment(shape=(18,12,12),dt=.001,calibrate=False,label='primary'):
    measurements=prepare_measurements(ROOT/'data'/'turbine_1L');points=observation_points()
    observed=measured_observations(measurements,points);provider=ExperimentalProvider(measurements)
    solver=IPCS(BOUNDS,shape,points,dt=dt);cache={};evaluations=[]
    def solve(nu):
        key=float(nu)
        if key not in cache:
            r=solver.solve(provider,measurements['times'][:10],key,verbose=True)
            pred=r['prediction'][1:];residual=velocity_residual(pred,observed[1:10],.4)
            cache[key]=r;evaluations.append({'nu':key,'J':float(residual@residual),'seconds':r['seconds']})
            save(f'{label}_progress',{'evaluations':evaluations,'status':'running'})
            np.savez_compressed(OUT/f'fit_latest_{label}.npz',nu=key,prediction=pred)
            print(label,evaluations[-1],flush=True)
        return cache[key]['prediction'][1:]
    nominal=solve(NU_REF);best=NU_REF;opt=None
    if calibrate:
        best,opt=fit_viscosity(solve,observed[1:10],NU_REF,.4,eta_bounds=(np.log(.5),np.log(2)),max_nfev=12)
    best_fit=solve(best);rfit=cache[float(best)]
    val=solver.solve(provider,measurements['times'][10:],best,verbose=True)
    rval_ref=val if best==NU_REF else solver.solve(provider,measurements['times'][10:],NU_REF)
    vpred=val['prediction'][1:];vref=rval_ref['prediction'][1:]
    profile=[]
    if calibrate:
        for nu in (NU_REF*.5,NU_REF,NU_REF*2):
            residual=velocity_residual(solve(nu),observed[1:10],.4)
            profile.append({'nu':nu,'J':float(residual@residual)})
    summary={'solver':'FEniCSx P2/P1 IPCS, CN/AB2 semi-implicit advection; no upwind/artificial viscosity',
       'dolfinx_version':rfit['dolfinx_version'],'grid_subdivisions':list(shape),'dt_s':dt,
       'velocity_dofs':rfit['velocity_dofs'],'pressure_dofs':rfit['pressure_dofs'],
       'nu_reference':NU_REF,'nu_fit':best,'calibrated':calibrate,'nu_bounds':[NU_REF*.5,NU_REF*2],
       'fit_at_bound':bool(best<NU_REF*.5005 or best>NU_REF*1.998),
       'optimizer_success':bool(opt.success) if opt is not None else None,
       'optimizer_message':opt.message if opt is not None else None,
       'fit_reference':velocity_metrics(nominal,observed[1:10]),'fit_calibrated':velocity_metrics(best_fit,observed[1:10]),
       'validation_reference':velocity_metrics(vref,observed[11:]),'validation_calibrated':velocity_metrics(vpred,observed[11:]),
       'component_relative_validation':np.sqrt(np.sum((vpred-observed[11:])**2,axis=(0,1))/np.sum(observed[11:]**2,axis=(0,1))).tolist(),
       'max_divergence_rms_s_inv':float(max(rfit['divergence_rms'].max(),val['divergence_rms'].max())),
       'max_divergence_reference_rms_s_inv':float(max(cache[float(NU_REF)]['divergence_rms'].max(),rval_ref['divergence_rms'].max())),
       'max_boundary_flux_m3_per_s':float(max(abs(rfit['boundary_flux']).max(),abs(val['boundary_flux']).max())),
       'max_boundary_normal_shift_m_per_s':float(max(abs(rfit['boundary_normal_shift']).max(),abs(val['boundary_normal_shift']).max())),
       'evaluations':evaluations,'profile':profile,
       'comparison_policy':'Same 18 real frames, 539 physical observation points, windows, interpolation and flux correction rule as MAC; separate projected IC for each discretization, p0=0',
       'weak_incompressibility':'P1 pressure / IPCS does not enforce pointwise zero divergence like the MAC projection'}
    save(label,summary)
    np.savez_compressed(OUT/f'solution_{label}.npz',times=measurements['times'],points=points,observed_points=observed,
                        fit_prediction=best_fit,validation_prediction=vpred,validation_reference_prediction=vref)
    save(f'{label}_progress',{'evaluations':evaluations,'status':'completed'})
    print('RESULT',json.dumps(summary,ensure_ascii=False),flush=True)
    solver.close();return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['check','verify','smoke','primary','coarser','finer','half_dt'],default='verify')
    args=parser.parse_args()
    if args.phase=='check':check()
    elif args.phase=='verify':verify()
    elif args.phase=='smoke':experiment((8,6,6),.001,False,'smoke')
    elif args.phase=='primary':experiment((18,12,12),.001,True,'primary')
    elif args.phase=='coarser':experiment((12,8,8),.001,False,'coarser')
    elif args.phase=='finer':experiment((27,18,18),.001,False,'finer')
    elif args.phase=='half_dt':experiment((18,12,12),.0005,False,'half_dt')
