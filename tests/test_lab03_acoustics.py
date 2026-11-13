"""Numerical and sampling invariants; independent of network and conda."""
import json
import numpy as np
import networkx as nx
from lab03.model import bits,graph,fit,flux_operators,relative_error,ROOT
from lab03.analytic import coefficients,evaluate
from lab03.solver import solve,verification
from lab03.sampling import sample_round,summary,error,transformed
from lab03.sampling import catalog,run_sampler


def test_all_expression_graphs_are_trees_and_dags():
    for mask in range(64):
        g=graph(mask)
        assert nx.is_directed_acyclic_graph(g)
        assert all(g.out_degree(n)==1 for n in g if n!='p_tt')
        assert g.number_of_edges()==g.number_of_nodes()-1
        for n in g:
            positions=[d['position'] for _,_,d in g.in_edges(n,data=True)]
            assert len(positions)==len(set(positions))


def test_analytic_wave_satisfies_equation_without_fd_derivatives():
    z=evaluate(coefficients(42,6),np.linspace(0,1,33),[0,.7,2.])
    assert relative_error(z['ptt'],.01*(z['xx']+z['yy']))<1e-13
    assert np.max(abs(z['pt'][0]))==0
    assert np.max(abs(z['p'][:,:,[0,-1]]))<1e-15


def test_rejection_draws_match_known_finite_target():
    rows=[{'J':j/10,'gate':j%3!=0} for j in range(12)]
    r=sample_round(rows,.5,n=150000,seed=42)
    assert abs(r['acceptance_rate']-r['expected_acceptance_rate'])<.005
    assert r['total_variation']<.015
    assert not np.isin(r['proposal'][r['accepted']],[0,3,6,9]).any()
    probability=np.array([x['gate']*np.exp(-x['J']/.5) for x in rows])
    assert np.array_equal(r['accepted'],r['uniform']<=probability[r['proposal']])


def test_verlet_spatial_temporal_convergence():
    r=verification()
    assert r['constant_field_error']==0
    assert all(3.5<q<4.8 for q in r['refinement_ratios'])


def test_linear_source_intervention_commutes_with_solver():
    x=np.linspace(0,1,33);p=np.sin(np.pi*x[:,None])*np.sin(np.pi*x[None,:])
    a,_=solve(p,1,.01,[0,.5,1],[1,1,0,.2,0,0],1/32,'dirichlet')
    b,_=solve(1.5*p,1,.01,[0,.5,1],[1,1,0,.2,0,0],1/32,'dirichlet')
    assert relative_error(1.5*a,b)<1e-6


def test_moment_score_equals_direct_residual():
    rng=np.random.default_rng(5);X=rng.normal(size=(500,6));p=rng.normal(size=6)
    y=rng.normal(size=500);d={'X':X,'y':y}
    assert abs(error(summary(d),p)-relative_error(y,X@p))<1e-12


def test_published_cno_endpoints():
    with np.load(ROOT/'data/cno_published_cases.npz') as d:
        x=np.linspace(0,1,64)
        for j in range(3):
            ref=d[f'case_{j}_output'];pred=evaluate(d[f'case_{j}_A'],x,[5.])['p'][0]
            assert relative_error(ref,pred)<1e-12


def test_saved_samples_obey_gate_and_graph_parameters():
    for name in ('well','analytic'):
        rows=json.loads((ROOT/'results'/f'{name}_catalog.json').read_text())
        with np.load(ROOT/'results'/f'{name}_sampling.npz') as d:
            for k in range(3):
                ids=d[f'r{k}_proposal'][d[f'r{k}_accepted']]
                assert np.array_equal(d[f'r{k}_reason']==0,d[f'r{k}_accepted'])
                assert all(rows[i]['gate'] for i in ids)
                assert len(ids)>0


def test_bootstrap_cannot_veto_original_physics():
    rng=np.random.default_rng(11);X=rng.normal(size=(200,6));y=X[:,0]+X[:,1]
    d={'X':X,'y':y}
    rule={'removable':[False]*6}
    forward={str(i):{'success':True,'pressure_error':.05,'intervention_error':.05} for i in range(64)}
    records=catalog(d,d,[d],rule,forward=forward)
    assert records[3]['gate']
    assert not records[3]['gate_failures']
    assert records[3]['bootstrap_risk']==4/6


def test_forward_error_and_complexity_beat_false_significance():
    rng=np.random.default_rng(12);X=rng.normal(size=(300,6));y=X[:,0]+X[:,1]+.5*X[:,3]
    d={'X':X,'y':y};rule={'removable':[False,False,True,False,True,True]}
    f={str(i):{'success':True,'pressure_error':.3,'intervention_error':.3} for i in range(64)}
    f['3']={'success':True,'pressure_error':.04,'intervention_error':.04}
    records=catalog(d,d,[d],rule,forward=f)
    assert records[3]['gate'] and records[11]['gate']
    assert records[3]['J']<records[11]['J']


def test_selection_refuses_incomplete_forward_validation():
    import pytest
    with pytest.raises(ValueError,match='Forward validation'):
        run_sampler([{'J':.1,'gate':True,'forward_verified':False}])


def test_modal_solver_against_independent_exact_formula():
    from lab03.modal_solver import solve_modes,exact_modes,relative_modal_error
    A=coefficients(99,8);t=np.linspace(0,5,31)
    assert relative_modal_error(solve_modes(A,t,[1,1,0,0,0,0]),exact_modes(A,t))<1e-8


def test_transfer_score_uses_separate_references_and_fixed_coefficients(monkeypatch):
    from lab03 import modal_solver as modal
    fields=[coefficients(seed,4) for seed in (100,101,102)]
    calls=[]
    theta=[1.,1.,0.,0.,0.,0.]
    def numerical(A,times,parameters,pressure_scale):
        index=next(i for i,z in enumerate(fields) if np.array_equal(A,z))
        calls.append((index,list(parameters),pressure_scale))
        return modal.exact_modes(A,times)*[1.,1.1,1.3][index]
    monkeypatch.setattr(modal,'solve_modes',numerical)
    result=modal.validation_catalog([{'mask':3,'parameters':theta,'physical_tensor':True}],
        fields[0],fields[1:],pressure_scale=17.)['3']
    assert np.allclose(result['transfer_errors'],[.1,.3])
    assert abs(result['intervention_error']-.2)<1e-12
    assert calls==[(i,theta,17.) for i in range(3)]


def test_feature_pressure_scale_is_fixed_across_initial_conditions(tmp_path):
    from lab03.model import well_features
    rng=np.random.default_rng(123)
    p=rng.normal(size=(31,32,32));rho=np.full((32,32),3.)
    results=[]
    for amplitude in (1.,2.):
        path=tmp_path/f'field_{amplitude}.npz'
        np.savez(path,p=amplitude*p,rho=rho,c=np.sqrt(4/rho),t=np.arange(31)*.02)
        results.append(well_features(path,pressure_scale=7.))
    a,b=results
    assert a['pressure_scale']==b['pressure_scale']==7.
    assert np.allclose(b['X'][:,:5],2*a['X'][:,:5])
    assert np.allclose(b['X'][:,5],8*a['X'][:,5])


def test_three_criteria_isolate_complexity_and_statistical_gate():
    from lab03.criterion_comparison import variant_catalog
    config={'moment_weight':.1,'pressure_weight':1.,'intervention_weight':.5,
        'bootstrap_weight':.01,'complexity_weight_per_branch':.02}
    rows=[{'mask':m,'gate':True,'gate_failures':[],'complexity':int(bits(m).sum()),
        'validation':.02,'validation_pressure':.1,'intervention_pressure':.2,
        'bootstrap_risk':0.,'J':.202+.02*int(bits(m).sum())} for m in range(64)]
    rule={'removable':[True]*6}
    current=variant_catalog(rows,rule,config,'current')
    unpenalized=variant_catalog(rows,rule,config,'no_complexity')
    assert [r['J'] for r in current]==[r['J'] for r in rows]
    assert unpenalized[3]['J']==unpenalized[63]['J']
    assert all(r['gate'] for r in unpenalized)
    rule['removable'][3]=False
    hard=variant_catalog(rows,rule,config,'hard_filter')
    assert not hard[3]['gate'] and 'required_branch_3' in hard[3]['gate_failures']
    assert hard[11]['gate']
    assert all(q['gate']==bool(bits(q['mask'])[3]) for q in hard)
    assert rows[3]['gate'] and rows[3]['gate_failures']==[]


def test_criterion_comparison_pairs_proposals_uniforms_and_temperatures():
    from lab03.criterion_comparison import paired_sampling
    a=[{'J':.1+i/100,'gate':True} for i in range(64)]
    b=[{'J':.2+i/100,'gate':i%2==0} for i in range(64)]
    ar,_=paired_sampling(a,[.4,.2,.1],123)
    br,_=paired_sampling(b,[.4,.2,.1],123)
    for x,y in zip(ar,br):
        assert x['T']==y['T']
        assert np.array_equal(x['proposal'],y['proposal'])
        assert np.array_equal(x['uniform'],y['uniform'])
        assert not np.array_equal(x['accepted'],y['accepted'])
