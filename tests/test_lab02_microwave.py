"""Numerical and statistical invariants, independent of committed results."""
import numpy as np
import pytest
from scipy.optimize import least_squares
from lab02.model import design, fit, predict, gamma, C, propagation_graph, expanded_maxwell_graph
from lab02.statistics import block_indices, bootstrap, distances, train_mask
from lab02.data import verify_sources
from lab02.upstream import ExpressionDAG, maxwell_model


def test_exact_identification_and_null_reduction():
    f=np.linspace(3e9,18e9,80);p=np.array([2.,.06,0.]);y=predict(f,p)
    for loss in ('linear','soft_l1'):
        np.testing.assert_allclose(fit(f,y,[.003,.008],loss),p,atol=1e-10)
    np.testing.assert_allclose(fit(f,y,[.003,.008],fixed={2:0.}),p,atol=1e-10)
    q=fit(f,y,[.003,.008],fixed={1:0.})
    assert np.linalg.norm(predict(f,q)-y)>0.01


def test_robust_solver_matches_independent_scipy_and_resists_outliers():
    f=np.linspace(3e9,18e9,100);truth=np.array([1.003,.06,0.]);y=predict(f,truth)
    y[::9,0] += .1
    scale=np.array([.003,.008]);a=(design(f)/scale[None,:,None]).reshape(-1,3)
    target=(y/scale).ravel()
    ref=least_squares(lambda p:a@p-target,truth,loss='soft_l1',ftol=1e-12,xtol=1e-12,gtol=1e-12).x
    robust=fit(f,y,scale,'soft_l1');ols=fit(f,y,scale)
    np.testing.assert_allclose(robust,ref,rtol=2e-5,atol=2e-6)
    assert np.linalg.norm(robust-truth)<np.linalg.norm(ols-truth)


def test_blocks_keep_frequency_adjacency_and_exclude_validation():
    mask=train_mask(151);length=8
    ids=block_indices(mask,length,np.random.default_rng(3))
    assert np.all(mask[ids])
    assert len(ids)==mask.sum()
    for start in range(0,len(ids)-length+1,length):
        np.testing.assert_array_equal(np.diff(ids[start:start+length]),1)
    with pytest.raises(ValueError):block_indices(np.zeros(10,bool),3,np.random.default_rng(1))


def test_zero_intervention_identity_and_reproducibility():
    f=np.linspace(3e9,18e9,40);y=predict(f,[2.,.06,0.]);mask=train_mask(40)
    z=np.random.default_rng(1).normal(size=y.shape);s=np.ones_like(y)*.001
    args=(f,y,mask,s,z)
    b,p=bootstrap(*args,n_boot=12,strengths=(0.,1.))
    b2,p2=bootstrap(*args,n_boot=12,strengths=(0.,1.))
    np.testing.assert_array_equal(p[:,0],np.broadcast_to(b,p[:,0].shape))
    np.testing.assert_array_equal(p,p2)
    assert np.std(p[2,1,:,1])>0
    raw,norm,joint=distances(b,b,np.ones(3),np.eye(3))
    assert np.all(raw==0) and np.all(norm==0) and joint==0


def test_lossless_gamma_and_harmonic_maxwell_identity():
    f=np.array([3e9,10e9]);theta=np.array([2.,.06,.004])
    g=gamma(f,theta);omega=2*np.pi*f
    eps=C*C*(g.imag**2-g.real**2)/omega**2
    sigma_over_eps0=2*C*C*g.real*g.imag/omega
    # i*w*eps*e + (sigma/eps0)*e = -c*dh/dx; h=c*gamma/(i*w)*e.
    lhs=1j*omega*eps+sigma_over_eps0
    rhs=C*C*g*g/(1j*omega)
    np.testing.assert_allclose(lhs,rhs,rtol=1e-14)
    np.testing.assert_allclose(gamma(f,[2.,0.,0.]),1j*omega*2/C)


def test_attributed_sources_and_student_graph_roundtrip():
    assert verify_sources()>=40
    for g in (propagation_graph(),maxwell_model(),maxwell_model(True),expanded_maxwell_graph(),expanded_maxwell_graph(False)):
        restored=ExpressionDAG.from_adjacency(g.adjacency(),g.to_dict())
        assert restored.to_dict()==g.to_dict()
