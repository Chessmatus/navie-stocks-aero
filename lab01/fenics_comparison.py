"""Read completed FEniCSx runs and compare with the preserved MAC calculation."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_comparison(directory):
    directory=Path(directory)
    mac=json.loads((directory/'primary.json').read_text())
    fem=json.loads((directory/'fenicsx'/'primary.json').read_text())
    verification=json.loads((directory/'fenicsx'/'verification.json').read_text())
    return mac,fem,verification


def discretization_report(directory):
    directory=Path(directory)/'fenicsx'
    runs={name:json.loads((directory/f'{name}.json').read_text())
          for name in ('coarser','primary','finer','half_dt')}
    solutions={name:np.load(directory/f'solution_{name}.npz') for name in runs}
    primary=solutions['primary']['validation_reference_prediction']
    half=solutions['half_dt']['validation_prediction']
    finer=solutions['finer']['validation_prediction']
    report={'nu_for_checks':runs['primary']['nu_reference'],
            'control_error_at_reference_nu':{name:r['validation_reference']['relative_l2'] for name,r in runs.items()},
            'halved_dt_field_difference_relative_l2':float(np.linalg.norm(primary-half)/np.linalg.norm(half)),
            'primary_vs_finer_field_difference_relative_l2':float(np.linalg.norm(primary-finer)/np.linalg.norm(finer)),
            'maximum_strong_divergence_of_reported_run_s_inv':{name:r['max_divergence_rms_s_inv'] for name,r in runs.items()},
            'interpretation':'Same physical observation points; differing initial projections. These checks do not establish convergence or pointwise incompressibility.'}
    (directory/'discretization_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return runs,report


def comparison_figures(directory):
    directory=Path(directory)
    fem=np.load(directory/'fenicsx'/'solution_primary.npz')
    mac=np.load(directory/'solution_primary.npz')
    points=fem['points'];observed=fem['observed_points'][11:]
    fp=fem['validation_prediction'];mp=mac['validation_prediction']
    if fp.shape!=mp.shape or not np.allclose(points,mac['points']):
        raise ValueError('Comparison requires identical physical points and times')
    t=fem['times'][11:]
    j=np.argmin(np.linalg.norm(points-[.06,0,0],axis=1))
    fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True,layout='constrained')
    for c,ax in enumerate(axes):
        ax.plot(t,observed[:,j,c],'ko-',label='Измерения',ms=4)
        ax.plot(t,mp[:,j,c],'s--',label='MAC',ms=4)
        ax.plot(t,fp[:,j,c],'d-',label='FEniCSx P2/P1',ms=4)
        ax.set_ylabel(f'{"uvw"[c]}, м/с');ax.grid(alpha=.25)
    axes[0].legend(ncol=3,fontsize=9)
    axes[0].set_title('Контрольное окно; x=60 мм, y=z=0')
    axes[-1].set_xlabel('t, с')
    fig.savefig(directory/'fenicsx'/'comparison_time_series.png',dpi=170)
    zmask=np.isclose(points[:,2],0)
    x=np.unique(points[zmask,0])*1000;y=np.unique(points[zmask,1])*1000
    X,Y=np.meshgrid(x,y,indexing='ij')
    measured=observed[-1,zmask,0].reshape(X.shape)
    fields=[measured,mp[-1,zmask,0].reshape(X.shape),fp[-1,zmask,0].reshape(X.shape)]
    fig2,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True,layout='constrained')
    lo=min(f.min() for f in fields);hi=max(f.max() for f in fields)
    for ax,field,title in zip(axes,fields,['Измерения','MAC','FEniCSx P2/P1']):
        im=ax.pcolormesh(X,Y,field,shading='nearest',vmin=lo,vmax=hi,cmap='viridis')
        ax.set_title(title);ax.set_xlabel('x, мм');ax.set_aspect('equal')
    axes[0].set_ylabel('y, мм');fig2.colorbar(im,ax=axes,label='u, м/с')
    fig2.suptitle('Сечение z=0; t=0.17 с; общие точки сравнения')
    fig2.savefig(directory/'fenicsx'/'comparison_slice.png',dpi=170)
    return {'time_series':fig,'slice':fig2}
