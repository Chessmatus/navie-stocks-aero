"""Run the complete lab; no network needed after fetching selected data."""
from pathlib import Path
import argparse
import json
import hashlib
import time
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter
from .model import ROOT,TERMS,bits,fit,well_features,relative_error,export_graph
from .analytic import feature_set,published_feature_set,archive_audit,save_coefficients,coefficients,evaluate
from .statistics import pruning,bootstrap
from .sampling import catalog,transformed,run_sampler,sample_round,error,summary
from .solver import solve,verification
from .modal_solver import validation_catalog,solve_modes,exact_modes,relative_modal_error


def clean_record(data):
    return {k:v for k,v in data.items() if k not in ('X','y','group','A')}


def experiment(name,train,valid,test,extra,n_boot):
    print('Inference',name,flush=True)
    rule=pruning(train,valid,n_boot=n_boot)
    interventions=[transformed(valid,1.5),transformed(valid,swap=True)]+extra
    preliminary=catalog(train,valid,interventions,rule)
    if name=='well':
        cache=ROOT/'results/well_forward_validation.json'
        payload=json.loads(cache.read_text()) if cache.exists() else {}
        forward=payload.get('records',{})
        need=[row for row in preliminary if row['physical_tensor'] and
              (str(row['mask']) not in forward or not np.allclose(
               forward[str(row['mask'])].get('parameters',[]),row['parameters'],atol=1e-9,rtol=1e-9))]
        if need:
            (ROOT/'results/validation_candidates.json').write_text(json.dumps(preliminary,indent=2))
            python=ROOT.parent/'.clawpack/bin/python'
            if not python.exists():raise RuntimeError('Missing Clawpack validation cache: create lab03/clawpack.yml environment')
            subprocess.run([str(python),str(ROOT/'clawpack_validation.py')],check=True)
            forward=json.loads(cache.read_text())['records']
    else:
        cache=ROOT/'results'/f'{name}_forward_validation.json'
        signature=hashlib.sha256(valid['A'].tobytes()+extra[0]['A'].tobytes()+
            np.array([row['parameters'] for row in preliminary]).tobytes()).hexdigest()
        saved=json.loads(cache.read_text()) if cache.exists() else {}
        if saved.get('signature')==signature:forward=saved['records']
        else:
            forward=validation_catalog(preliminary,valid['A'],extra[0]['A'])
            cache.write_text(json.dumps({'signature':signature,'records':forward},indent=2))
    records=catalog(train,valid,interventions,rule,forward=forward)
    rounds,selected=run_sampler(records)
    parameters=np.asarray(records[selected]['parameters'])
    # Fresh test evaluation starts only after the graph is fixed.
    test=test() if callable(test) else test
    np.savez_compressed(ROOT/'results'/f'{name}_sampling.npz',
        **{f'r{k}_{key}':value for k,r in enumerate(rounds) for key,value in r.items()})
    (ROOT/'results'/f'{name}_catalog.json').write_text(json.dumps(records,indent=2))
    export_graph(selected,parameters,ROOT/'results'/name)
    export_graph(63,rule['full_parameters'],ROOT/'results'/f'{name}_full')
    baseline=bootstrap(train,mask=selected,n_boot=n_boot,block=8)
    np.savez_compressed(ROOT/'results'/f'{name}_bootstrap.npz',parameters=baseline)
    selected_record={**records[selected], 'test_moment_error':relative_error(test['y'],test['X']@parameters),
        'ci95_selected':np.quantile(baseline,[.025,.975],axis=0).tolist()}
    ungated=min((r for r in records if r['gate']),key=lambda r:r['J'])
    # Selection comparisons under the same fixed rule and budget.
    no_int=catalog(train,valid,interventions,rule,use_interventions=False,forward=forward)
    _,without=run_sampler(no_int)
    repeats=[]
    for seed in range(10):
        rr,ss=run_sampler(records,seed=7000+seed)
        repeats.append({'seed':7000+seed,'selected':ss,
            'acceptance_rates':[r['acceptance_rate'] for r in rr]})
    exact_best=min((r for r in records if r['gate']),key=lambda r:r['J'])['mask']
    # Independent audit with fixed T, enough accepted draws to inspect the pmf.
    audit=sample_round(records,T=rounds[0]['T'],n=100000,seed=99001)
    np.savez_compressed(ROOT/'results'/f'{name}_distribution_audit.npz',**audit)
    roundinfo=[{k:v for k,v in r.items() if k not in ('proposal','uniform','accepted','reason','expected_pmf','empirical_pmf')}
        | {'attempts':len(r['proposal']),'accepted_count':int(r['accepted'].sum())} for r in rounds]
    result={'selected':selected_record,'rule':rule,'rounds':roundinfo,
        'input':{'train':clean_record(train),'validation':clean_record(valid),'test':clean_record(test)},
        'comparison':{'selected_without_interventions':without,
            'ungated_best':ungated,'exhaustive_best_under_gate':exact_best,
            'repeats':repeats,'audit_TV':audit['total_variation'],
            'audit_acceptance_rate':audit['acceptance_rate'],
            'audit_expected_acceptance_rate':audit['expected_acceptance_rate']},
        'original_physics':{'mask':3,'parameters':[1.,1.,0.,0.,0.,0.],
            'test_moment_error':relative_error(test['y'],test['X']@np.array([1.,1.,0.,0.,0.,0.]))}}
    print(name,'mask',selected,'parameters',parameters,'test moments',selected_record['test_moment_error'],flush=True)
    return result


def run(n_boot=1000):
    started=time.monotonic()
    ROOT.joinpath('results').mkdir(exist_ok=True);save_coefficients()
    report={'seed':20261112,'method':'Rejection sampling (method 4)',
        'data_status':'Published simulations and exact analytical benchmark; no experimental measurements; 2D+time.',
        'candidate_terms':list(TERMS),'candidate_count':64,
        'criterion':json.loads((ROOT/'criterion.json').read_text()),
        'evaluation':{'well_test_sample':1,'analytic_test_seed':55001,
            'old_well_test_0':'Development-only after criterion v1 was inspected'},
        'verification':verification()}
    tr=well_features(ROOT/'data/well_train_0.npz')
    va=well_features(ROOT/'data/well_valid_0.npz')
    report['well']=experiment('well',tr,va,lambda:well_features(ROOT/'data/well_test_1.npz'),[],n_boot)
    report['analytic_archive_audit']=archive_audit()
    report['analytic']=experiment('analytic',published_feature_set(0),published_feature_set(1),
        lambda:feature_set(55001,32,.85),[feature_set(3104)],n_boot)
    report['analytic_noise']=experiment('analytic_noise',published_feature_set(0,noise=.01),published_feature_set(1),
        lambda:feature_set(55001,32,.85),[feature_set(3104)],n_boot)
    p=np.array(report['analytic']['selected']['parameters'])
    report['analytic']['ood_moment_error']=error(summary(feature_set(3105,32,.85)),p)
    print('PDE rollouts',flush=True)
    with np.load(ROOT/'data/well_test_1.npz') as d:
        p0=d['p'][0];rho=d['rho'];bulk=np.median(d['rho']*d['c']**2)
        # All test frames are untouched until model and thresholds have been fixed.
        times=d['t'].astype(float)[::2];reference=d['p'][::2]
    rolls={}
    for label,theta in [('original',[1.,1.,0.,0.,0.,0.]),
                        ('selected',report['well']['selected']['parameters'])]:
        pred,steps=solve(p0,rho,bulk,times,theta,2/p0.shape[0])
        smooth_ref=gaussian_filter(reference.astype(float),(0,2.,2.))
        smooth_pred=gaussian_filter(pred.astype(float),(0,2.,2.))
        report['well'][f'{label}_rollout']={'relative_L2':relative_error(reference,pred),
            'lowpass_relative_L2':relative_error(smooth_ref,smooth_pred),
            'final_relative_L2':relative_error(reference[-1],pred[-1]),'steps':steps}
        rolls[label]=pred
        print(label,report['well'][f'{label}_rollout'],flush=True)
    np.savez_compressed(ROOT/'results/well_rollout.npz',times=times,reference=reference,
        rho=rho,**rolls)
    # Physical source-amplitude intervention, solved independently on the same grid.
    theta=report['well']['selected']['parameters']
    perturbed,_=solve(1.5*p0,rho,bulk,times,theta,2/p0.shape[0],pressure_scale=float(np.max(abs(p0))))
    report['well']['source_amplitude_covariance_error']=relative_error(1.5*rolls['selected'],perturbed)
    x=np.linspace(0,1,129);times=np.linspace(0,5,51)
    A=coefficients(55001,32,.85)
    exact=evaluate(A,x,times)['p']
    modal=solve_modes(A,times,p);S=np.sin(x[:,None]*np.pi*np.arange(1,len(A)+1))
    pred=np.asarray([S@z@S.T for z in modal]);steps=None
    original_modes=solve_modes(A,times,[1,1,0,0,0,0])
    original=np.asarray([S@z@S.T for z in original_modes])
    report['analytic']['selected_rollout']={'relative_L2':relative_error(exact,pred),
        'final_relative_L2':relative_error(exact[-1],pred[-1]),'steps':steps,
        'solver':'Sine-Galerkin + DOP853, rtol=1e-9'}
    report['analytic']['original_rollout']={'relative_L2':relative_error(exact,original)}
    np.savez_compressed(ROOT/'results/analytic_rollout.npz',times=times,x=x,
        reference=exact,selected=pred,original=original)
    # Forward comparisons on the frozen fresh test; no test-dependent refit.
    (ROOT/'results/report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    python=ROOT.parent/'.clawpack/bin/python'
    for model in ('original','selected'):
        path=ROOT/'results'/f'clawpack_test1_{model}.npz'
        expected=np.array([1.,1.,0.,0.,0.,0.] if model=='original' else report['well']['selected']['parameters'])
        valid_cache=False
        if path.exists():
            with np.load(path) as d:valid_cache=(np.allclose(d['theta'],expected,atol=1e-10,rtol=1e-10)
                and int(d['sample'])==1)
        if not valid_cache:
            if not python.exists():raise RuntimeError('Create the isolated Clawpack environment to run the new test')
            subprocess.run([str(python),str(ROOT/'clawpack_runner.py'),'--model',model,'--sample','1'],check=True)
    report['runtime_seconds']=time.monotonic()-started
    report['data_hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'data').glob('*.npz'))}
    attach_clawpack(report)
    (ROOT/'results/report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('Completed in',report['runtime_seconds'],'seconds',flush=True)
    return report


def attach_clawpack(report):
    with np.load(ROOT/'data/well_test_1.npz') as d:ref=d['p'][::2]
    for model in ('original','selected'):
        path=ROOT/'results'/f'clawpack_test1_{model}.npz'
        if path.exists():
            with np.load(path) as d:p=d['p']
            report['well'][f'clawpack_{model}']={
                'relative_L2':relative_error(ref,p),
                'final_relative_L2':relative_error(ref[-1],p[-1]),
                'solver':'Clawpack 5.9.0, unsplit MC, CFL 0.9'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args();run(args.bootstrap)
