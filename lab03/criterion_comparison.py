"""Paired comparison of three selection criteria on the frozen v3 data.

Only N weight and the statistical gate vary. All coefficients and forward
validation errors are reused unchanged. Test sample 2 is a post-hoc diagnostic,
not a newly blinded test. Proposal masks, uniforms and temperatures are paired.
"""
from pathlib import Path
import copy,json,hashlib,subprocess
import numpy as np
from .model import ROOT,bits,TERMS,export_graph
from .sampling import sample_round
from .modal_solver import solve_modes,exact_modes,relative_modal_error
from .analytic import coefficients

CASES=(('current','1. Текущий критерий'),('no_complexity','2. Без штрафа за сложность'),
       ('hard_filter','3. Без сложности + исходный фильтр'))
OUT=ROOT/'results/criterion_comparison'


def variant_catalog(rows,rule,config,case):
    if case not in dict(CASES):raise ValueError(case)
    answer=copy.deepcopy(rows)
    for row in answer:
        row['base_gate']=row['gate']
        row['statistical_gate']=all(bits(row['mask'])[j] or removable
            for j,removable in enumerate(rule['removable']))
        if case=='hard_filter' and not row['statistical_gate']:
            row['gate']=False
            row['gate_failures']+=['required_branch_'+str(j) for j,r in enumerate(rule['removable'])
                if not bits(row['mask'])[j] and not r]
        # No subtraction of .02N: avoid cancellation near machine precision.
        if case!='current':
            row['J']=(config['moment_weight']*row['validation']+
                config['bootstrap_weight']*row['bootstrap_risk']+
                config['pressure_weight']*(row['validation_pressure'] or 0.)+
                config['intervention_weight']*(row['intervention_pressure'] or 0.))
        row['complexity_weight']=config['complexity_weight_per_branch'] if case=='current' else 0.
    return answer


def paired_sampling(rows,temperatures,seed):
    rounds=[sample_round(rows,float(T),seed=seed+k) for k,T in enumerate(temperatures)]
    ids=set(np.concatenate([r['proposal'][r['accepted']] for r in rounds]).tolist())
    if not ids:raise RuntimeError('No accepted graph')
    # No implicit preference for N, including in exact ties.
    selected=min(ids,key=lambda i:(rows[i]['J'],i))
    return rounds,selected


def run_comparison():
    OUT.mkdir(exist_ok=True)
    base=json.loads((ROOT/'results/report.json').read_text());cfg=base['criterion']
    assert cfg['version']==3
    datasets={};plan={'test_sample':base['evaluation']['well_test_sample'],
        'pressure_scale':base['well']['input']['train']['pressure_scale'],'models':{}}
    for name in ('well','analytic','analytic_noise'):
        source=ROOT/'results'/f'{name}_catalog.json';rows=json.loads(source.read_text())
        rule=base[name]['rule'];temperatures=[r['T'] for r in base[name]['rounds']]
        cases={}
        for key,label in CASES:
            records=variant_catalog(rows,rule,cfg,key)
            rounds,selected=paired_sampling(records,temperatures,base['seed'])
            np.savez_compressed(OUT/f'{name}_{key}_sampling.npz',
                **{f'r{k}_{field}':value for k,r in enumerate(rounds) for field,value in r.items()})
            (OUT/f'{name}_{key}_catalog.json').write_text(json.dumps(records,indent=2))
            eligible=[q for q in records if q['gate']]
            best=min(eligible,key=lambda q:(q['J'],q['mask']))['mask']
            repeats=[]
            for seed in range(7000,7010):
                _,chosen=paired_sampling(records,temperatures,seed);repeats.append(chosen)
            q=records[selected]
            cases[key]={'label':label,'selected':q,'terms':[t for j,t in enumerate(TERMS) if bits(selected)[j]],
                'allowed_count':len(eligible),'exhaustive_best':best,'repeat_selections':repeats,
                'rounds':[{'T':r['T'],'attempts':len(r['proposal']),'accepted':int(r['accepted'].sum()),
                    'acceptance_rate':r['acceptance_rate']} for r in rounds]}
            export_graph(selected,q['parameters'],OUT/f'{name}_{key}')
            if name=='well':plan['models'][str(selected)]=q['parameters']
        datasets[name]={'catalog_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'required_terms':[t for t,r in zip(TERMS,rule['removable']) if not r],
            'temperatures':temperatures,'cases':cases}
    # Every graph is fixed before the comparison's test evaluation starts.
    (OUT/'selection.json').write_text(json.dumps({'datasets':datasets,'well_forward_plan':plan},indent=2,ensure_ascii=False))
    subprocess.run([str(ROOT.parent/'.clawpack/bin/python'),str(ROOT/'criterion_comparison_forward.py')],check=True)
    with np.load(ROOT/'data'/f"well_test_{plan['test_sample']}.npz") as d:ref=d['p'][::2]
    for q in datasets['well']['cases'].values():
        path=OUT/f"well_mask{q['selected']['mask']}_test{plan['test_sample']}.npz"
        with np.load(path) as d:pred=d['p']
        q['test_pressure_error']=float(np.linalg.norm(pred.astype(float)-ref)/np.linalg.norm(ref.astype(float)))
        q['test_final_error']=float(np.linalg.norm(pred[-1].astype(float)-ref[-1])/np.linalg.norm(ref[-1].astype(float)))
    A=coefficients(base['evaluation']['analytic_test_seed'],32,.85);times=np.linspace(0,5,51)
    reference=exact_modes(A,times)
    for name in ('analytic','analytic_noise'):
        scale=base[name]['input']['train']['pressure_scale']
        for q in datasets[name]['cases'].values():
            q['test_pressure_error']=relative_modal_error(
                solve_modes(A,times,q['selected']['parameters'],pressure_scale=scale),reference)
    report={'protocol':{'comparison':'Post-hoc criterion ablation on the same v3 data and fitted coefficients',
        'seed':base['seed'],'attempts_per_round':5000,'rounds':3,
        'paired_proposals':True,'paired_uniforms':True,'paired_temperatures':True,
        'complexity_weight':[.02,0.,0.],'bootstrap_soft_weight':cfg['bootstrap_weight'],
        'hard_gate':[False,False,True],'test_status':'Previously inspected sample_2 / seed55002: diagnostic comparison, not a fresh blind test',
        'test_well_sample':plan['test_sample'],'test_analytic_seed':base['evaluation']['analytic_test_seed'],
        'criterion_sha256':hashlib.sha256((ROOT/'criterion.json').read_bytes()).hexdigest()},'datasets':datasets}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    from .comparison_report import build_comparison_report
    build_comparison_report(report)
    return report


if __name__=='__main__':
    report=run_comparison()
    for name,data in report['datasets'].items():
        for key,q in data['cases'].items():
            print(name,key,'mask',q['selected']['mask'],'N',q['selected']['complexity'],
                'base',q['selected']['validation_pressure'],'transfer',q['selected']['intervention_pressure'],
                'test',q['test_pressure_error'],flush=True)
