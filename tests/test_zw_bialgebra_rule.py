import pytest

from pyzx.graph.graph_s import GraphS
from pyzx.utils import VertexType, EdgeType
from pyzx.tensor import tensorfy, compare_tensors
from pyzx.rewrite_rules.zw_bialgebra_rule import (
    check_zw_bialgebra, unsafe_zw_bialgebra, zw_bialgebra,
)

S = EdgeType.SIMPLE
H = EdgeType.HADAMARD


def _add_w(g, q, r):
    w_in = g.add_vertex(VertexType.W_INPUT, q, r)
    w_out = g.add_vertex(VertexType.W_OUTPUT, q, r + 0.3)
    g.add_edge((w_in, w_out), EdgeType.W_IO)
    return w_in, w_out


def build_compact(in_edges, out_edges):
    """inputs -> Z(phase 0) -> W -> outputs. in_edges/out_edges give each
    leg's EdgeType."""
    g = GraphS()
    n, m = len(in_edges), len(out_edges)
    ins = [g.add_vertex(VertexType.BOUNDARY, i, 0) for i in range(n)]
    z = g.add_vertex(VertexType.Z, n, 1)
    w_in, w_out = _add_w(g, n, 2)
    outs = [g.add_vertex(VertexType.BOUNDARY, i, 4) for i in range(m)]
    for b, et in zip(ins, in_edges):
        g.add_edge((b, z), et)
    g.add_edge((z, w_in), S)
    for b, et in zip(outs, out_edges):
        g.add_edge((w_out, b), et)
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g, z, w_in


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_expand_preserves_tensor(n, m):
    g, z, w_in = build_compact([S] * n, [S] * m)
    before = tensorfy(g)
    assert check_zw_bialgebra(g, z, w_in)
    assert unsafe_zw_bialgebra(g, z, w_in)
    assert compare_tensors(before, tensorfy(g), preserve_scalar=True)


def test_expand_order_insensitive():
    g, z, w_in = build_compact([S, S], [S, S])
    assert check_zw_bialgebra(g, w_in, z)  # swapped order still matches


def test_expand_preserves_hadamard_outer_legs():
    g, z, w_in = build_compact([H, S], [S, H])
    before = tensorfy(g)
    assert zw_bialgebra(g, z, w_in)
    assert compare_tensors(before, tensorfy(g), preserve_scalar=True)


def test_expand_rejects_nonzero_phase():
    g, z, w_in = build_compact([S, S], [S])
    g.set_phase(z, 1)  # pi phase
    assert not check_zw_bialgebra(g, z, w_in)


def test_expand_rejects_z_on_w_output_side():
    # Z glued to a W_OUTPUT leg is NOT a valid redex.
    g = GraphS()
    w_in, w_out = _add_w(g, 0, 0)
    z = g.add_vertex(VertexType.Z, 0, 1)
    b = g.add_vertex(VertexType.BOUNDARY, 0, 2)
    g.add_edge((w_out, z), S)
    g.add_edge((z, b), S)
    g.add_edge((w_in, g.add_vertex(VertexType.BOUNDARY, 1, -1)), S)
    assert not check_zw_bialgebra(g, z, w_out)
    assert not check_zw_bialgebra(g, z, w_in)


def test_expand_rejects_extra_leg_on_w_input():
    g, z, w_in = build_compact([S, S], [S])
    extra = g.add_vertex(VertexType.BOUNDARY, 5, 5)
    g.add_edge((w_in, extra), S)  # W_INPUT now has a second external leg
    assert not check_zw_bialgebra(g, z, w_in)


def test_expand_rejects_no_input_legs():
    # A phase-0 Z whose only leg is the edge to W_INPUT (n=0) is not a valid
    # redex: expanding it with m>=2 outputs is not a tensor identity.
    g, z, w_in = build_compact([], [S, S])
    assert not check_zw_bialgebra(g, z, w_in)


from pyzx.rewrite_rules.zw_bialgebra_rule import (
    match_zw_bialgebra_op, unsafe_zw_bialgebra_op,
    safe_apply_zw_bialgebra_op, is_zw_bialg_op_match,
)


def build_bipartite(n, m):
    """inputs -> W...W -> Z...Z -> outputs (complete bipartite). Returns the
    graph and the list of interior vertices (all W halves + Z spiders)."""
    g = GraphS()
    ins = [g.add_vertex(VertexType.BOUNDARY, i, 0) for i in range(n)]
    outs = [g.add_vertex(VertexType.BOUNDARY, i, 4) for i in range(m)]
    ws = [_add_w(g, i, 1) for i in range(n)]
    zs = [g.add_vertex(VertexType.Z, j, 3) for j in range(m)]
    for b, (w_in, w_out) in zip(ins, ws):
        g.add_edge((b, w_in), S)
    for b, z in zip(outs, zs):
        g.add_edge((z, b), S)
    for (w_in, w_out) in ws:
        for z in zs:
            g.add_edge((w_out, z), S)
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    interior = [v for pair in ws for v in pair] + zs
    return g, interior


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("m", [2, 3])
def test_reduce_preserves_tensor(n, m):
    g, interior = build_bipartite(n, m)
    before = tensorfy(g)
    assert is_zw_bialg_op_match(g, interior)
    assert safe_apply_zw_bialgebra_op(g, interior)
    assert compare_tensors(before, tensorfy(g), preserve_scalar=True)


def test_reduce_then_expand_roundtrip():
    g, interior = build_bipartite(2, 3)
    before = tensorfy(g)
    matches = match_zw_bialgebra_op(g, interior)
    assert matches is not None
    assert unsafe_zw_bialgebra_op(g, matches)
    # Now compact: find the Z and W_INPUT and expand again.
    z = next(v for v in g.vertices() if g.type(v) == VertexType.Z)
    w_in = next(v for v in g.vertices() if g.type(v) == VertexType.W_INPUT)
    assert unsafe_zw_bialgebra(g, z, w_in)
    assert compare_tensors(before, tensorfy(g), preserve_scalar=True)


def test_reduce_rejects_incomplete_bipartite():
    g, interior = build_bipartite(2, 2)
    # Drop one bipartite edge so it is no longer complete.
    w_out = next(v for v in g.vertices() if g.type(v) == VertexType.W_OUTPUT)
    z = next(v for v in g.vertices() if g.type(v) == VertexType.Z)
    for e in list(g.edges(w_out, z)):
        g.remove_edge(e)
    assert match_zw_bialgebra_op(g, interior) is None


def test_reduce_rejects_internal_output_leg():
    # A bipartite-looking region whose Z "output" legs connect to each other
    # (in-region) rather than to outside vertices must be rejected: otherwise
    # the applier would wire an edge to a just-deleted vertex.
    g = GraphS()
    ins = [g.add_vertex(VertexType.BOUNDARY, i, 0) for i in range(2)]
    ws = [_add_w(g, i, 1) for i in range(2)]
    zs = [g.add_vertex(VertexType.Z, j, 3) for j in range(2)]
    for b, (w_in, w_out) in zip(ins, ws):
        g.add_edge((b, w_in), S)
    for (w_in, w_out) in ws:
        for z in zs:
            g.add_edge((w_out, z), S)
    g.add_edge((zs[0], zs[1]), S)   # Z "output" legs point at each other (in-region)
    g.set_inputs(tuple(ins))
    interior = [v for pair in ws for v in pair] + zs
    assert match_zw_bialgebra_op(g, interior) is None


def test_simplify_exposes_zw_objects():
    import pyzx.simplify as simplify
    from pyzx.rewrite import RewriteSimpDoubleVertex, RewriteSimpGraph
    assert isinstance(simplify.zw_bialg_simp, RewriteSimpDoubleVertex)
    assert isinstance(simplify.zw_bialg_op_simp, RewriteSimpGraph)

    # zw_bialg_simp.apply expands a compact redex.
    g, z, w_in = build_compact([S, S], [S, S])
    before = tensorfy(g)
    assert simplify.zw_bialg_simp.is_match(g, z, w_in)
    assert simplify.zw_bialg_simp.apply(g, z, w_in)
    assert compare_tensors(before, tensorfy(g), preserve_scalar=True)

    # zw_bialg_op_simp round-trips it back.
    interior = [v for v in g.vertices()
                if g.type(v) in (VertexType.W_INPUT, VertexType.W_OUTPUT,
                                 VertexType.Z)]
    assert simplify.zw_bialg_op_simp.is_match(g, interior)
    assert simplify.zw_bialg_op_simp.apply(g, interior)
    assert compare_tensors(before, tensorfy(g), preserve_scalar=True)


def test_reduce_rejects_internal_input_leg():
    # A W node whose "input" leg points at another matched vertex (in-region)
    # rather than an outside vertex must be rejected.
    g = GraphS()
    ws = [_add_w(g, i, 1) for i in range(2)]
    zs = [g.add_vertex(VertexType.Z, j, 3) for j in range(2)]
    outs = [g.add_vertex(VertexType.BOUNDARY, j, 4) for j in range(2)]
    for (w_in, w_out) in ws:
        for z in zs:
            g.add_edge((w_out, z), S)
    for z, b in zip(zs, outs):
        g.add_edge((z, b), S)
    # Point W-node 0's input leg at W-node 1's input (in-region) instead of outside.
    g.add_edge((ws[0][0], ws[1][0]), S)
    g.set_outputs(tuple(outs))
    interior = [v for pair in ws for v in pair] + zs
    assert match_zw_bialgebra_op(g, interior) is None
