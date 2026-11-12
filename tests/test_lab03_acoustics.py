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
