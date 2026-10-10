"""Run real-data 3D NS calibration, temporal validation and discretization checks."""
from pathlib import Path
import json
import time
import numpy as np
from lab01.solver import MACGrid,solve_window
from lab01.experiment import (BOUNDS,prepare_measurements,MeasuredBoundary,
                             observation_points,measured_observations,observe_solution)
from lab01.calibration import fit_viscosity,velocity_metrics,velocity_residual

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'turbine_1L';OUT.mkdir(parents=True,exist_ok=True)
NU_REF=.4*.045/20000
NU_BOUNDS=(.5*NU_REF,2*NU_REF)


def run_configuration(measurements,points,observed,shape,cfl=.35,label='primary',profile=False):
    grid=MACGrid(BOUNDS,shape);boundary=MeasuredBoundary(grid,measurements)
    fit_times=measurements['times'][:10];val_times=measurements['times'][10:]
    cache={};evaluations=[]
    def solve(nu):
        key=float(nu)
        if key not in cache:
            start=time.perf_counter();r=solve_window(grid,boundary,fit_times,key,cfl=cfl)
            prediction=observe_solution(grid,r['velocity'],points)[1:]
            cache[key]=(prediction,r)
            residual=velocity_residual(prediction,observed[:10][1:],.4)
            evaluations.append({'nu':key,'J':float(residual@residual),'steps':r['steps'],
                                'seconds':time.perf_counter()-start})
            print(f'{label} nu={key:.8g}, J={evaluations[-1]["J"]:.6g}',flush=True)
        return cache[key][0]
    best,result=fit_viscosity(solve,observed[:10][1:],NU_REF,.4,
                              eta_bounds=tuple(np.log(np.array(NU_BOUNDS)/NU_REF)),max_nfev=20)
    nominal=solve(NU_REF);best_prediction=solve(best)
    best_fit=cache[float(best)][1]
    profile_values=[]
    if profile:
        for nu in np.geomspace(*NU_BOUNDS,9):
            residual=velocity_residual(solve(float(nu)),observed[:10][1:],.4)
            profile_values.append({'nu':float(nu),'J':float(residual@residual)})
    val_nominal=solve_window(grid,boundary,val_times,NU_REF,cfl=cfl)
    val_best=solve_window(grid,boundary,val_times,best,cfl=cfl)
    hv_nominal=observe_solution(grid,val_nominal['velocity'],points)[1:]
    hv_best=observe_solution(grid,val_best['velocity'],points)[1:]
    summary={'label':label,'grid_cells':list(shape),'spacing_m':list(grid.h),'cfl':cfl,
       'nu_reference':NU_REF,'nu_bounds':list(NU_BOUNDS),'nu_fit':best,
       'fit_at_bound':bool(best/NU_BOUNDS[0]<1.001 or best/NU_BOUNDS[1]>.999),
       'optimizer_success':bool(result.success),'optimizer_message':result.message,
       'optimizer_nfev':int(result.nfev),
       'fit_reference':velocity_metrics(nominal,observed[:10][1:]),
       'fit_calibrated':velocity_metrics(best_prediction,observed[:10][1:]),
       'validation_reference':velocity_metrics(hv_nominal,observed[10:][1:]),
       'validation_calibrated':velocity_metrics(hv_best,observed[10:][1:]),
       'max_divergence_rms_s_inv':float(max(best_fit['divergence_rms'].max(),val_best['divergence_rms'].max())),
       'max_boundary_normal_shift_m_per_s':float(max(abs(best_fit['boundary_normal_shift']).max(),abs(val_best['boundary_normal_shift']).max())),
       'profile':profile_values,'evaluations':evaluations}
    np.savez_compressed(OUT/f'predictions_{label}.npz',fit=best_prediction,validation=hv_best)
    if profile:
        np.savez_compressed(OUT/'solution_primary.npz',
            x=grid.centers[0],y=grid.centers[1],z=grid.centers[2],
            times=measurements['times'],points=points,observed_points=observed,
            fit_prediction=best_prediction,validation_prediction=hv_best,
            validation_reference_prediction=hv_nominal,
            fit_velocity=best_fit['velocity'],validation_velocity=val_best['velocity'],
            observed_grid=boundary.observed,
            fit_divergence=best_fit['divergence_rms'],validation_divergence=val_best['divergence_rms'])
    (OUT/f'{label}.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(f'{label}: nu={best:.8g}, fit={summary["fit_calibrated"]["relative_l2"]:.3%}, '
          f'validation={summary["validation_calibrated"]["relative_l2"]:.3%}',flush=True)
    return summary


def main():
    measurements=prepare_measurements(ROOT/'data'/'turbine_1L')
    points=observation_points();observed=measured_observations(measurements,points)
    runs=[]
    for shape,cfl,label,profile in [((36,24,24),.35,'primary',True),
                                   ((24,16,16),.35,'coarser',False),
                                   ((48,32,32),.35,'finer',False),
                                   ((36,24,24),.175,'half_time_step',False)]:
        runs.append(run_configuration(measurements,points,observed,shape,cfl,label,profile))
    fields={r['label']:dict(np.load(OUT/f"predictions_{r['label']}.npz")) for r in runs}
    reference=fields['finer']
    discretization={label:{window:velocity_metrics(fields[label][window],reference[window])['relative_l2'] for window in ('fit','validation')} for label in ('coarser','primary')}
    discretization['time_step_change']={window:velocity_metrics(fields['primary'][window],fields['half_time_step'][window])['relative_l2'] for window in ('fit','validation')}
    persistence={'fit':velocity_metrics(np.broadcast_to(observed[0],observed[1:10].shape),observed[1:10]),
                 'validation':velocity_metrics(np.broadcast_to(observed[10],observed[11:].shape),observed[11:])}
    report={'dataset_doi':'10.18710/SX0ZQT','case':'1L',
      'frame_ids':measurements['frame_ids'].tolist(),'times_s':measurements['times'].tolist(),
      'domain_bounds_m':BOUNDS,'observation_points':len(points),
      'fit_frames':measurements['frame_ids'][:10].tolist(),
      'validation_frames':measurements['frame_ids'][10:].tolist(),
      'solver':'3D MAC upwind advection / centered diffusion + projected SSP-RK3 + DCT pressure Poisson',
      'boundary_policy':'Measured time-dependent Dirichlet velocity; uniform outward-normal flux correction',
      'comparison_policy':'Same 539 interior physical observation points for all grids; initial frames excluded',
      'nu_search_policy':'Exploratory factor-two interval around the Re-derived reference; not a measured temperature constraint',
      'not_claimed':'DNS resolution, independent raw-PTV validation, autonomous forecasting, or reliable molecular viscosity identification',
      'discretization_changes_relative_l2':discretization,'persistence_baseline':persistence,
      'runs':runs}
    (OUT/'experiment_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('Saved real experimental calculation to outputs/turbine_1L',flush=True)
    return report


if __name__=='__main__':main()
