"""Verify the selected tree and the time-dependent analytical 3D field."""
import networkx as nx
import numpy as np
import sympy as sp

from lab01.graphs import vorticity_tree, OperatorDAG
from lab01.symbolic import symbolic_equations
from lab01.visualization import travelling_abc_symbolic, travelling_abc_field


def test_tree_matches_3d_vorticity_rhs_and_survives_export():
    tree = vorticity_tree()
    graph = nx.DiGraph(tree.graph)
    assert nx.is_tree(graph)
    assert len(graph) == 11 and graph.number_of_edges() == 10
    assert all(graph.out_degree(n) == (0 if n == "n11" else 1) for n in graph)
    restored = OperatorDAG.from_dict(tree.to_dict())
    assert restored.to_dict() == tree.to_dict()
    order, z, _ = tree.matrices()
    assert order == [f"n{i:02}" for i in range(1, 12)]
    assert not np.count_nonzero(np.triu(z))
    assert np.flatnonzero(z[-1]).tolist() == [4, 9]

    eq = symbolic_equations()
    x, y, zcoord, _ = eq["coordinates"]
    coords = (x, y, zcoord)
    velocity = sp.Matrix([y*zcoord, zcoord*x, x*x])
    def curl(a):
        j = a.jacobian(coords)
        return sp.Matrix([j[2, 1]-j[1, 2], j[0, 2]-j[2, 0], j[1, 0]-j[0, 1]])
    operators = {"gradient": lambda a: a.jacobian(coords), "curl": curl,
                 "matvec": lambda j, a: j*a, "subtract": lambda a, b: a-b}
    leaves = {n: velocity for n, d in graph.nodes(data=True) if d["op"] is None}
    actual = restored.evaluate(leaves, operators)["n11"]
    expected = eq["inviscid_rhs"].subs(dict(zip(eq["u"], velocity))).doit()
    assert actual != sp.zeros(3, 1)
    assert (actual-expected).applyfunc(sp.simplify) == sp.zeros(3, 1)


def test_travelling_abc_solves_full_3d_equations_with_time():
    eq = symbolic_equations()
    x, y, z, t = eq["coordinates"]
    for kappa in (sp.S.Zero, sp.Rational(1, 20)):
        field = travelling_abc_symbolic(kappa=kappa)
        u = field["u"]
        substitutions = dict(zip(eq["u"], u))
        substitutions.update({eq["p"]: field["p"], eq["rho"]: 1, eq["nu"]: kappa})
        rhs = eq["momentum_rhs"].subs(substitutions).doit()
        assert (sp.diff(u, t)-rhs).applyfunc(sp.trigsimp) == sp.zeros(3, 1)
        assert eq["divergence"].subs(substitutions).doit().simplify() == 0
        assert (eq["omega"].subs(substitutions).doit()-field["omega"]).applyfunc(sp.simplify) == sp.zeros(3, 1)
        # Compare plotted arrays with the independently differentiated symbolic field.
        numeric = travelling_abc_field(0.4, 0.9, 1.1, 1.3, kappa=float(kappa))
        expected = np.array(u.subs({x:.4, y:.9, z:1.1, t:1.3}), dtype=float).ravel()
        np.testing.assert_allclose(numeric, expected, rtol=1e-13, atol=1e-13)
    assert not np.allclose(travelling_abc_field(.4, .9, 1.1, 0),
                           travelling_abc_field(.4, .9, 1.1, 2))
