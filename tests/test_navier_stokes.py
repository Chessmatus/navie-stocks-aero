"""Check the current 16-node tree, projection and a known 3D PDE solution."""
import networkx as nx
import numpy as np
import sympy as sp
from lab01.graphs import navier_stokes_tree, OperatorDAG
from lab01.solver import MACGrid, corrected_boundary, project, solve_window


def test_original_momentum_tree_and_dictionary():
    tree=navier_stokes_tree();g=nx.DiGraph(tree.graph)
    assert nx.is_tree(g) and len(g)==16 and g.number_of_edges()==15
    assert all(g.out_degree(n)==(0 if n=='n16' else 1) for n in g)
    tree=OperatorDAG.from_dict(tree.to_dict())
    order,z,_=tree.matrices()
    assert order==[f'n{i:02}' for i in range(1,17)]
    assert not np.count_nonzero(np.triu(z))
    x,y,zcoord,t=sp.symbols('x y z t');coords=(x,y,zcoord)
    u=sp.Matrix([y*zcoord+t,zcoord*x,x*x]);p=x*y+zcoord**2+t*x
    nu,rho=sp.Rational(1,10),sp.Rational(3,2)
    lap=lambda a:a.applyfunc(lambda f:sum(sp.diff(f,c,2) for c in coords))
    ops={'gradient':lambda a:a.jacobian(coords) if isinstance(a,sp.MatrixBase) else sp.Matrix([sp.diff(a,c) for c in coords]),
         'matvec':lambda a,b:a*b,'negate':lambda a:-a,'divide_scalar':lambda a,b:a/b,
         'add':lambda a,b:a+b,'laplacian':lap,'scale':lambda a,b:a*b}
    values={'n01':u,'n03':u,'n13':u,'n06':p,'n08':rho,'n12':nu}
    actual=tree.evaluate(values,ops)['n16']
    expected=-u.jacobian(coords)*u-sp.Matrix([sp.diff(p,c)/rho for c in coords])+nu*lap(u)
    assert (actual-expected).applyfunc(sp.simplify)==sp.zeros(3,1)


def test_projection_preserves_normal_boundary_and_removes_divergence():
    grid=MACGrid(((0,1),(0,2),(0,3)),(7,6,5));rng=np.random.default_rng(42)
    faces=[rng.normal(size=grid.face_shape(c)) for c in range(3)]
    boundary,shift=corrected_boundary(grid,faces)
    state,pressure=project(grid,faces,boundary,.02)
    assert np.isfinite(shift) and abs(pressure.mean())<1e-10
    assert np.max(abs(grid.divergence(state)))<1e-12
    for c in range(3):
        for side in (0,-1):
            np.testing.assert_array_equal(np.take(state[c],side,axis=c),np.take(boundary[c],side,axis=c))


from lab01.verification import abc_error


def test_3d_pde_error_decreases_with_refinement():
    coarse,div_coarse=abc_error(12,.005)
    fine,div_fine=abc_error(24,.0025)
    assert fine<.8*coarse and fine<.025
    assert max(div_coarse,div_fine)<1e-12
