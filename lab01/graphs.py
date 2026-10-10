"""Typed operator DAGs. Edges go from operands to operations.

The laboratory convention is Z[target, source] = 1. Argument ports,
including repeated operands, are stored separately from binary adjacency.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np


class OperatorDAG:
    """Explicit node IDs distinguish equal labels from shared subexpressions."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def leaf(self, node_id, label, kind="field", shape="vector3"):
        if node_id in self.graph:
            raise ValueError(f"Duplicate node: {node_id}")
        self.graph.add_node(node_id, label=label, kind=kind, shape=shape, op=None)
        return node_id

    def operation(self, node_id, label, op, args, shape="vector3"):
        if node_id in self.graph:
            raise ValueError(f"Duplicate node: {node_id}")
        if any(arg not in self.graph for arg in args):
            raise ValueError("Operands must be defined before their operation")
        self.graph.add_node(node_id, label=label, kind="operation", shape=shape, op=op)
        for port, arg in enumerate(args):
            self.graph.add_edge(arg, node_id, key=port, port=port)
        return node_id

    def ordered_args(self, node_id):
        edges = self.graph.in_edges(node_id, keys=True, data=True)
        return [src for src, _, _, data in sorted(edges, key=lambda edge: edge[3]["port"])]

    def matrices(self):
        """Return deterministic topological order, binary Z and count matrix M."""
        order = list(nx.lexicographical_topological_sort(self.graph, key=str))
        counts = nx.to_numpy_array(self.graph, nodelist=order, weight=None, dtype=int).T
        return order, (counts > 0).astype(int), counts

    def to_dict(self):
        """Dictionary is lossless: metadata and ordered inputs accompany topology."""
        order, _, _ = self.matrices()
        return {"schema": "operator-dag-v1", "edge_direction": "input_to_operation",
                "matrix_convention": "Z[target,source]", "order": order, "graph": dict(self.graph.graph),
                "nodes": {n: {**self.graph.nodes[n], "args": self.ordered_args(n)} for n in order}}

    @classmethod
    def from_dict(cls, payload):
        if payload["schema"] != "operator-dag-v1":
            raise ValueError("Unknown graph schema")
        dag = cls()
        dag.graph.graph.update(payload.get("graph", {}))
        for n in payload["order"]:
            item = dict(payload["nodes"][n])
            args = item.pop("args")
            dag.graph.add_node(n, **item)
            for port, src in enumerate(args):
                if src not in dag.graph:
                    raise ValueError("Invalid topological order")
                dag.graph.add_edge(src, n, key=port, port=port)
        return dag

    def evaluate(self, values, operators):
        """Evaluate with explicitly supplied differential operators / BC policy.

        Graph structure alone cannot define differentiation or a Poisson solver.
        """
        results = dict(values)
        for n in nx.topological_sort(self.graph):
            op = self.graph.nodes[n]["op"]
            if op is None:
                if n not in results:
                    raise KeyError(f"Missing input {n}")
            else:
                results[n] = operators[op](*(results[a] for a in self.ordered_args(n)))
        return results


def vorticity_dag(viscous=True):
    """RHS = stretching - transport + kappa*laplacian(omega).

    omega = curl(u). Setting viscous=False removes diffusion, matching the
    inviscid evolution in Scarano et al. (2022), Eq. (2). This is an expression
    DAG, not the complete VIC-TSA algorithm or a numerical PDE solver.
    """
    d = OperatorDAG()
    d.leaf("u", "u(x,y,z,t)")
    d.operation("omega", "curl(u)", "curl", ["u"])
    d.operation("grad_u", "grad(u)", "gradient", ["u"], "matrix3x3")
    d.operation("grad_omega", "grad(omega)", "gradient", ["omega"], "matrix3x3")
    # Convention J[i,j] = partial_j u_i; matrix-vector multiplication gives (a.grad)u.
    d.operation("stretch", "(omega.grad)u", "matvec", ["grad_u", "omega"])
    d.operation("transport", "(u.grad)omega", "matvec", ["grad_omega", "u"])
    d.operation("inviscid", "stretch - transport", "subtract", ["stretch", "transport"])
    if viscous:
        d.leaf("kappa", "kappa=1/Re", kind="parameter", shape="scalar")
        d.operation("lap_omega", "laplacian(omega)", "laplacian", ["omega"])
        d.operation("diffusion", "kappa * lap(omega)", "scale", ["kappa", "lap_omega"])
        d.operation("rhs", "d(omega)/dt", "add", ["inviscid", "diffusion"])
    else:
        d.operation("rhs", "d(omega)/dt", "identity", ["inviscid"])
    d.operation("div_u", "div(u)=0", "divergence", ["u"], "scalar")
    return d


def vorticity_tree():
    """Expression tree for the selected inviscid RHS, with slide IDs 1..11.

    Every occurrence of u and curl(u) is separate. The root computes
    grad(u) @ curl(u) - grad(curl(u)) @ u. No constraint or solver is
    attached to this connected expression tree.
    """
    d = OperatorDAG()
    d.leaf("n01", "u")
    d.operation("n02", "grad", "gradient", ["n01"], "matrix3x3")
    d.leaf("n03", "u")
    d.operation("n04", "curl", "curl", ["n03"])
    d.operation("n05", "mv", "matvec", ["n02", "n04"])
    d.leaf("n06", "u")
    d.operation("n07", "curl", "curl", ["n06"])
    d.operation("n08", "grad", "gradient", ["n07"], "matrix3x3")
    d.leaf("n09", "u")
    d.operation("n10", "mv", "matvec", ["n08", "n09"])
    d.operation("n11", "−", "subtract", ["n05", "n10"])
    d.graph.graph.update(output="n11", representation="expression_tree")
    return d


def navier_stokes_tree():
    """Original momentum RHS; IDs and argument order match the presentation.

    F = -(grad(u)) @ u - grad(p)/rho + nu*laplacian(u).
    The pressure and incompressibility constraint belong to the PDE solver,
    not to this expression tree. Repeated velocity inputs are separate leaves.
    """
    d = OperatorDAG()
    d.leaf("n01", "u")
    d.operation("n02", "grad", "gradient", ["n01"], "matrix3x3")
    d.leaf("n03", "u")
    d.operation("n04", "mv", "matvec", ["n02", "n03"])
    d.operation("n05", "−", "negate", ["n04"])
    d.leaf("n06", "p", shape="scalar")
    d.operation("n07", "grad", "gradient", ["n06"])
    d.leaf("n08", "ρ", kind="parameter", shape="scalar")
    d.operation("n09", "÷s", "divide_scalar", ["n07", "n08"])
    d.operation("n10", "−", "negate", ["n09"])
    d.operation("n11", "+", "add", ["n05", "n10"])
    d.leaf("n12", "ν", kind="parameter", shape="scalar")
    d.leaf("n13", "u")
    d.operation("n14", "Δ", "laplacian", ["n13"])
    d.operation("n15", "×s", "scale", ["n12", "n14"])
    d.operation("n16", "+", "add", ["n11", "n15"])
    d.graph.graph.update(output="n16", representation="expression_tree",
                         equation="original_incompressible_momentum_rhs")
    return d


def plot_tree(tree):
    """Draw the selected expression tree with the root above its operands."""
    import matplotlib.pyplot as plt
    graph = nx.DiGraph(tree.graph)
    if not nx.is_tree(graph) or max(dict(graph.out_degree()).values()) > 1:
        raise ValueError("Expected a connected expression tree, inputs -> operations")
    if tree.graph.graph.get("equation") == "original_incompressible_momentum_rhs":
        positions = {"n16": (0, 0), "n11": (-2.2, -1), "n15": (3.6, -1),
                     "n05": (-4.1, -2), "n10": (-.35, -2), "n12": (2.5, -2),
                     "n14": (4.7, -2), "n04": (-4.1, -3), "n09": (-.35, -3),
                     "n13": (4.7, -3), "n02": (-5, -4), "n03": (-3.2, -4),
                     "n07": (-1.3, -4), "n08": (.6, -4), "n01": (-5, -5),
                     "n06": (-1.3, -5)}
        title = r"$F=-(\nabla\mathbf{u})\mathbf{u}-\nabla p/\rho+\nu\Delta\mathbf{u}$"
        legend = "mv: матрица × вектор; ÷s: вектор / скаляр; ×s: скаляр × вектор; стрелки: вход → операция"
    else:
        positions = {"n11": (0, 0), "n05": (-2.6, -1), "n10": (2.6, -1),
                     "n02": (-3.9, -2), "n04": (-1.3, -2), "n08": (1.3, -2),
                     "n09": (3.9, -2), "n01": (-3.9, -3), "n03": (-1.3, -3),
                     "n07": (1.3, -3), "n06": (1.3, -4)}
        title = "Дерево правой части: grad(u) @ curl(u) − grad(curl(u)) @ u"
        legend = "mv(M, v) = Mv; стрелки: вход → операция; листья u — одно и то же поле"
    labels = {n: f"{int(n[1:])}: {graph.nodes[n]['label']}" for n in graph}
    colors = [{"field": "#d8eaf9", "parameter": "#ffe0b3", "operation": "#d9edd9"}
              [graph.nodes[n]["kind"]] for n in graph]
    fig, ax = plt.subplots(figsize=(12.5, 7.2), layout="constrained")
    nx.draw_networkx(graph, positions, labels=labels, node_color=colors,
                     node_size=1900, node_shape="s", edgecolors="#666666",
                     font_size=11, arrowsize=17, ax=ax)
    ax.set_title(title)
    ax.text(0.5, -0.03, legend,
            transform=ax.transAxes, ha="center", fontsize=10)
    ax.margins(x=.13, y=.15)
    ax.axis("off")
    return fig


def export_dag(dag, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    order, binary, counts = dag.matrices()
    (directory / f"{stem}.json").write_text(json.dumps(dag.to_dict(), indent=2), encoding="utf-8")
    (directory / f"{stem}_nodes.json").write_text(json.dumps(order, indent=2), encoding="utf-8")
    for suffix, matrix in [("adjacency", binary), ("multiplicity", counts)]:
        np.savetxt(directory / f"{stem}_{suffix}.csv", matrix, fmt="%d", delimiter=",")


def sympy_graph(expression, share=True):
    """Convert a scalar SymPy expression to a tree or a shared-expression DAG.

    This reflects SymPy's canonical argument order, not the original LaTeX.
    Differential expressions should keep their srepr metadata for interpretation.
    """
    import sympy as sp
    from sympy.core.function import AppliedUndef
    dag = OperatorDAG()
    seen = {}

    def visit(expr):
        if share and expr in seen:
            return seen[expr]
        if expr.is_Atom or isinstance(expr, AppliedUndef):
            node_id = f"n{len(dag.graph)}"
            kind = "constant" if expr.is_number else "field"
            dag.leaf(node_id, str(expr), kind=kind, shape="scalar")
        else:
            args = [visit(arg) for arg in expr.args]
            node_id = f"n{len(dag.graph)}"
            dag.operation(node_id, expr.func.__name__, expr.func.__name__, args, "scalar")
        dag.graph.nodes[node_id]["sympy_srepr"] = sp.srepr(expr)
        seen[expr] = node_id
        return node_id

    dag.graph.graph["output"] = visit(expression)
    return dag


def plot_dag(dag):
    import matplotlib.pyplot as plt
    graph = nx.DiGraph(dag.graph)
    layers = list(nx.topological_generations(graph))
    pos = {n: (depth, (len(layer)-1)/2 - k) for depth, layer in enumerate(layers) for k, n in enumerate(layer)}
    colors = [{"field": "#65a9e8", "parameter": "#ffb14e", "constant": "#ffb14e", "operation": "#9bd29b"}[graph.nodes[n]["kind"]] for n in graph]
    fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
    labels = {n: graph.nodes[n]["label"] for n in graph}
    labels.update({"grad_omega": "grad\n(omega)", "lap_omega": "laplacian\n(omega)",
                   "stretch": "(omega.grad)\nu", "transport": "(u.grad)\nomega",
                   "inviscid": "stretch\n- transport", "diffusion": "kappa *\nlap(omega)"})
    labels = {n: labels[n] for n in graph}
    nx.draw_networkx(graph, pos=pos, labels=labels,
                     node_color=colors, node_size=3100, font_size=9, arrowsize=18, ax=ax)
    # A node can be used by multiple operations; mark shared intermediates.
    shared = [n for n in graph if graph.nodes[n]["kind"] == "operation" and graph.out_degree(n) > 1]
    nx.draw_networkx_nodes(graph, pos, nodelist=shared, node_size=3100, node_color="none", edgecolors="#cf3975", linewidths=2.5, ax=ax)
    ax.set_title("Vorticity equation: operands -> operations; pink = shared intermediate")
    ax.margins(x=.14, y=.2)
    ax.axis("off")
    return fig


def plot_adjacency(dag):
    import matplotlib.pyplot as plt
    order, matrix, _ = dag.matrices()
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=70, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("j: input / source")
    ax.set_ylabel("i: operation / target")
    ax.set_title("Z[i,j] = 1 if j is an input to i (laboratory convention)")
    return fig
