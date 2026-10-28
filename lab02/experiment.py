"""Reproducible experiments, no plotting or notebook state required."""
from pathlib import Path
import json
import platform
import time
import numpy as np
from .data import ROOT, measured, verify_sources
from .model import predict, fit, gamma, propagation_graph, expanded_maxwell_graph, material_node_map
from .statistics import train_mask, conditional_scale, bootstrap, distances, ablation
from .upstream import plane_waves, benchmark, ParameterVector, fit_one_parameter, maxwell_model

SEED = 20261021
STRENGTHS = np.array([0., .5, 1., 2., 4.])
VARIABLES = ('alpha','beta','both')
LOSSES = ('linear','soft_l1')
TRUTH = np.array([2., .06, 0.])


def synthetic():
    f = np.linspace(3e9, 18e9, 151)
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(len(f),2))
    rho = .6
    for i in range(1,len(z)):
        z[i] = rho*z[i-1]+np.sqrt(1-rho*rho)*z[i]
    scale = np.array([.002, .006])[None, :]*(.8+.4*(f/f.max()))[:, None]
    return f, predict(f, TRUTH)+scale*z


def upstream_check():
    rng = np.random.default_rng(73)
    points = rng.uniform(0,.03,(32,3))
    times = np.linspace(0,8e-11,7)
    def forward(eps):
        fields = [plane_waves(points, t, eps) for t in times]
        return tuple(np.array([field[i] for field in fields]) for i in range(2))
    fit_result = fit_one_parameter(forward, forward(4.), ParameterVector(
        ('epsilon_r',),(3.,),('1',),((2.,6.),)), tolerance=1e-7)
    convergence = [benchmark(n) for n in (12,24)]
    return {'epsilon_fit':fit_result['value'], 'truth':4., 'loss':fit_result['loss'],
            'fdtd_convergence':convergence,
            'electric_error_ratio':convergence[0]['electric_relative_l2']/convergence[1]['electric_relative_l2']}


def run(n_boot=1000, blocks=(8,4,12), rebuild_data=False):
    start = time.time()
    data = measured(rebuild_data)
    f_real = data['ZNA_frequency_hz']
    g_real = data['ZNA_gamma']
    datasets = {'synthetic': synthetic(), 'measured': (f_real, np.column_stack((g_real.real,g_real.imag)))}
    projection_rng = np.random.default_rng(532)
    directions = projection_rng.normal(size=(128,3))
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    arrays = {'strengths':STRENGTHS, 'directions':directions}
    report = {'seed':SEED,'bootstrap_replicates':n_boot,'block_length':blocks[0],
              'variables':VARIABLES,'losses':LOSSES,'strengths':STRENGTHS.tolist(),
              'data_source':'ZNA, 10 measured tuner offsets, Hatab et al. (2023)',
              'model':'gamma=a sqrt(f/f0)+b f/f0 + i(2 pi f n/c + a sqrt(f/f0))',
              'pruning_rule':{'confidence':.95,'rms_cost_tolerance':.25,'W1_SD_at_lambda1_max':1.},
              'source_files_verified':verify_sources(), 'python':platform.python_version(),
              'upstream_check':upstream_check(), 'datasets':{}}
    for name,(f,y) in datasets.items():
        print('Dataset:',name,flush=True)
        mask = train_mask(len(f))
        scale,z = conditional_scale(f,y,mask)
        p = fit(f[mask],y[mask],scale[mask])
        pred = predict(f,p)
        residual = y-pred
        info = {'n_frequency':len(f),'n_train':int(mask.sum()),'n_validation':int((~mask).sum()),
                'frequency_GHz':[float(f[0]*1e-9),float(f[-1]*1e-9)],
                'parameters':p.tolist(),'epsilon_asymptotic':float(p[0]**2),
                'validation_rmse_alpha_beta':np.sqrt(np.mean(residual[~mask]**2,axis=0)).tolist(),
                'train_rmse_alpha_beta':np.sqrt(np.mean(residual[mask]**2,axis=0)).tolist(),
                'loss_results':{},'block_sensitivity':{}}
        if name == 'synthetic':info['truth'] = TRUTH.tolist()
        for key,value in {'f':f,'y':y,'mask':mask,'scale':scale,'residual_z':z}.items():arrays[name+'_'+key]=value
        for loss in LOSSES:
            print('  bootstrap',loss,flush=True)
            base,samples = bootstrap(f,y,mask,scale,z,n_boot,blocks[0],SEED,loss,STRENGTHS,VARIABLES)
            normalizer = np.maximum(base.std(axis=0,ddof=1),1e-12)
            raw = np.zeros((len(VARIABLES),len(STRENGTHS),3))
            norm = raw.copy();sw=np.zeros(raw.shape[:2])
            for v in range(len(VARIABLES)):
                for l in range(len(STRENGTHS)):
                    raw[v,l],norm[v,l],sw[v,l] = distances(base,samples[v,l],normalizer,directions)
            prefix = name+'_'+loss
            for key,value in {'baseline':base,'samples':samples,'W1':raw,'W1_SD':norm,'SW1':sw,'normalizer':normalizer}.items():arrays[prefix+'_'+key]=value
            # Monte Carlo floor only; not a data-level hypothesis-test threshold.
            rng = np.random.default_rng(SEED+91)
            null = []
            for _ in range(100):
                a=base[rng.integers(0,n_boot,n_boot)];b=base[rng.integers(0,n_boot,n_boot)]
                null.append(distances(a,b,normalizer,directions)[2])
            info['loss_results'][loss] = {'fit':fit(f[mask],y[mask],scale[mask],loss).tolist(),
                'ci95':np.quantile(base,[.025,.975],axis=0).tolist(),
                'mean':base.mean(axis=0).tolist(),'std':normalizer.tolist(),
                'W1_SD_both_lambda1':norm[2,2].tolist(),'SW1_both_lambda1':float(sw[2,2]),
                'SW1_empirical_MC_floor95':float(np.quantile(null,.95)),
                'covariance':np.cov(base.T).tolist()}
            if loss == 'linear':
                decisions=ablation(f,y,mask,scale,base,block=blocks[0],n_boot=n_boot,stability=norm[2,2])
                info['ablation']=decisions
                # Distribution / coefficient decisions under block length variation.
                for length in blocks[1:]:
                    print('  block sensitivity',length,flush=True)
                    bb,_=bootstrap(f,y,mask,scale,z,n_boot,length,SEED,loss,(0.,),('both',))
                    arrays[f'{name}_block{length}_baseline']=bb
                    # Validation bands have 8 consecutive rows; use length<=8 there.
                    dd=ablation(f,y,mask,scale,bb,block=length,n_boot=n_boot,stability=norm[2,2])
                    info['block_sensitivity'][str(length)]={'ci95':np.quantile(bb,[.025,.975],axis=0).tolist(),
                        'decisions':[r['decision'] for r in dd]}
        # One concrete intervention realization for displaying data, never used as validation truth.
        from .statistics import block_indices
        rng=np.random.default_rng(SEED+17)
        noise=z[block_indices(mask,blocks[0],rng,size=len(f))]*scale
        arrays[name+'_intervention_example']=y+2*noise
        report['datasets'][name]=info
        final=[]
        for j,record in enumerate(info['ablation']):
            choices=[record['decision']]+[d['decisions'][j] for d in info['block_sensitivity'].values()]
            final.append({'parameter':record['parameter'],
                          'decision':'remove' if all(c=='remove' for c in choices) else 'keep',
                          'block_length_sensitive':len(set(choices))>1})
        info['final_decisions']=final
    # Other analyzers were neither optimized nor used to choose the model.
    theta = np.array(report['datasets']['measured']['parameters'])
    cross={}
    for instrument in ('ENA','VectorStar'):
        f=data[instrument+'_frequency_hz'];g=data[instrument+'_gamma'];p=gamma(f,theta)
        cross[instrument]={'n':len(f),'frequency_GHz':[float(f[0]*1e-9),float(f[-1]*1e-9)],
            'rmse_alpha_beta':[float(np.sqrt(np.mean((p.real-g.real)**2))),float(np.sqrt(np.mean((p.imag-g.imag)**2)))]}
        arrays[instrument+'_f']=f;arrays[instrument+'_gamma']=g
    report['cross_instrument']=cross
    report['elapsed_seconds']=time.time()-start
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'parameter_graph_map.json').write_text(json.dumps(material_node_map(),indent=2)+'\n')
    np.savez_compressed(out/'bootstrap_samples.npz',**arrays)
    for name,g in [('propagation',propagation_graph()),('student_lossless',maxwell_model()),('student_lossy',maxwell_model(True))]:
        (out/f'{name}_graph.json').write_text(json.dumps(g.to_dict(),indent=2)+'\n')
        np.save(out/f'{name}_adjacency.npy',g.adjacency())
    for name,info in report['datasets'].items():
        g=propagation_graph(include_extra=info['final_decisions'][2]['decision']!='remove')
        (out/f'{name}_selected_graph.json').write_text(json.dumps(g.to_dict(),indent=2)+'\n')
        np.save(out/f'{name}_selected_adjacency.npy',g.adjacency())
        expanded=expanded_maxwell_graph(include_extra=info['final_decisions'][2]['decision']!='remove')
        (out/f'{name}_maxwell_expanded_graph.json').write_text(json.dumps(expanded.to_dict(),indent=2)+'\n')
        np.save(out/f'{name}_maxwell_expanded_adjacency.npy',expanded.adjacency())
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def load():
    report=json.loads((ROOT/'results/report.json').read_text())
    with np.load(ROOT/'results/bootstrap_samples.npz') as d: arrays={k:d[k] for k in d.files}
    return report, arrays
