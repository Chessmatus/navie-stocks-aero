import networkx as nx
import numpy as np
import sympy as sp

from lab01.graphs import OperatorDAG, vorticity_dag, sympy_graph
from lab01.symbolic import symbolic_equations


def test_matrix_direction_and_topological_structure():
    dag = vorticity_dag()
    order, z, _ = dag.matrices()
    assert nx.is_directed_acyclic_graph(dag.graph)
    idx = {n: i for i, n in enumerate(order)}
    assert z[idx["omega"], idx["u"]] == 1
    assert z[idx["u"], idx["omega"]] == 0
    assert np.count_nonzero(np.triu(z)) == 0


def test_repeated_inputs_and_argument_order_survive_dictionary():
    d = OperatorDAG()
    d.leaf("x", "x", shape="scalar")
    d.leaf("y", "y", shape="scalar")
    d.operation("square", "x*x", "multiply", ["x", "x"], "scalar")
    d.operation("subtract", "y-x*x", "subtract", ["y", "square"], "scalar")
    restored = OperatorDAG.from_dict(d.to_dict())
    assert restored.to_dict() == d.to_dict()
    order, z, m = restored.matrices()
    assert z[order.index("square"), order.index("x")] == 1
    assert m[order.index("square"), order.index("x")] == 2
    result = restored.evaluate({"x": 3, "y": 4}, {"multiply": lambda a,b:a*b, "subtract": lambda a,b:a-b})
    assert result["subtract"] == -5


def test_curl_removes_pressure_for_smooth_fields():
    x, y, z = sp.symbols("x y z")
    p = sp.Function("p")(x, y, z)
    grad = sp.Matrix([sp.diff(p, c) for c in (x, y, z)])
    assert sp.diff(grad[2], y)-sp.diff(grad[1], z) == 0
    assert sp.diff(grad[0], z)-sp.diff(grad[2], x) == 0
    assert sp.diff(grad[1], x)-sp.diff(grad[0], y) == 0


def test_viscous_vorticity_rhs_on_3d_abc_field():
    # ABC Beltrami field has curl(u)=u, div(u)=0, lap(u)=-u.
    # Stretching and transport cancel, leaving -nu*u; inviscid RHS is zero.
    eq = symbolic_equations()
    x, y, z, t = eq["coordinates"]
    abc = sp.Matrix([sp.sin(z)+sp.cos(y), sp.sin(x)+sp.cos(z), sp.sin(y)+sp.cos(x)])
    substitution = dict(zip(eq["u"], abc))
    assert sp.simplify(eq["divergence"].subs(substitution).doit()) == 0
    visc = eq["vorticity_rhs"].subs(substitution).doit().applyfunc(sp.simplify)
    invisc = eq["inviscid_rhs"].subs(substitution).doit().applyfunc(sp.simplify)
    assert visc == -eq["nu"]*abc
    assert invisc == sp.zeros(3, 1)


def test_tree_and_shared_dag_compute_same_scalar_expression():
    x = sp.Symbol('x')
    expr = sp.sin(x) + sp.log(x)*x
    graphs = [sympy_graph(expr, share=share) for share in (False, True)]
    assert len(graphs[1].graph) < len(graphs[0].graph)
    operators = {"sin": np.sin, "log": np.log, "Add": lambda *a: sum(a), "Mul": lambda *a: np.prod(a)}
    for dag in graphs:
        values = {n: 2.0 for n, attr in dag.graph.nodes(data=True) if attr['op'] is None}
        result = dag.evaluate(values, operators)[dag.graph.graph['output']]
        assert np.isclose(result, float(expr.subs(x, 2)))
