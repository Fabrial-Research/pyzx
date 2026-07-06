import pytest
from pyzx.graph import Graph
from pyzx.utils import (VertexType, EdgeType, vertex_is_triangle,
                        vertex_is_triangle_input, vertex_is_triangle_output,
                        get_triangle_partner, get_triangle_io)
from pyzx import tikz as zxtikz


def _inverse_graph():
    g = Graph('multigraph')
    g.set_auto_simplify(False)
    tip = g.add_vertex(VertexType.TRIANGLE_INVERSE_INPUT, 0, 1)
    body = g.add_vertex(VertexType.TRIANGLE_INVERSE_OUTPUT, 0, 2)
    g.add_edge((tip, body), EdgeType.W_IO)
    return g, tip, body


def test_inverse_type_values():
    assert int(VertexType.TRIANGLE_INVERSE_INPUT) == 9
    assert int(VertexType.TRIANGLE_INVERSE_OUTPUT) == 10


def test_vertex_is_triangle_covers_inverse():
    assert vertex_is_triangle(VertexType.TRIANGLE_INVERSE_INPUT)
    assert vertex_is_triangle(VertexType.TRIANGLE_INVERSE_OUTPUT)
    # existing triangle still recognised
    assert vertex_is_triangle(VertexType.TRIANGLE_INPUT)
    assert vertex_is_triangle(VertexType.TRIANGLE_OUTPUT)
    assert not vertex_is_triangle(VertexType.W_OUTPUT)


def test_triangle_role_helpers():
    assert vertex_is_triangle_input(VertexType.TRIANGLE_INPUT)
    assert vertex_is_triangle_input(VertexType.TRIANGLE_INVERSE_INPUT)
    assert not vertex_is_triangle_input(VertexType.TRIANGLE_OUTPUT)
    assert vertex_is_triangle_output(VertexType.TRIANGLE_OUTPUT)
    assert vertex_is_triangle_output(VertexType.TRIANGLE_INVERSE_OUTPUT)
    assert not vertex_is_triangle_output(VertexType.TRIANGLE_INPUT)


def test_inverse_partner_and_io():
    g, tip, body = _inverse_graph()
    assert get_triangle_partner(g, tip) == body
    assert get_triangle_partner(g, body) == tip
    assert get_triangle_io(g, tip) == (tip, body)
    assert get_triangle_io(g, body) == (tip, body)


import numpy as np


def _inverse_map(input_first: bool):
    """1-in-1-out graph containing a single triangle-inverse.
    input_first=True lays tip before body; False lays body before tip."""
    g = Graph('multigraph')
    g.set_auto_simplify(False)
    inp = g.add_vertex(VertexType.BOUNDARY, 0, 0)
    if input_first:
        tip = g.add_vertex(VertexType.TRIANGLE_INVERSE_INPUT, 0, 1)
        body = g.add_vertex(VertexType.TRIANGLE_INVERSE_OUTPUT, 0, 2)
    else:
        body = g.add_vertex(VertexType.TRIANGLE_INVERSE_OUTPUT, 0, 1)
        tip = g.add_vertex(VertexType.TRIANGLE_INVERSE_INPUT, 0, 2)
    outp = g.add_vertex(VertexType.BOUNDARY, 0, 3)
    g.add_edge((inp, tip), EdgeType.SIMPLE)
    g.add_edge((tip, body), EdgeType.W_IO)
    g.add_edge((body, outp), EdgeType.SIMPLE)
    g.set_inputs((inp,))
    g.set_outputs((outp,))
    return g


def test_inverse_matrix():
    m = _inverse_map(input_first=True).to_matrix()
    assert np.allclose(m, np.array([[1, -1], [0, 1]]))


def test_inverse_matrix_layout_independent():
    m = _inverse_map(input_first=False).to_matrix()
    assert np.allclose(m, np.array([[1, -1], [0, 1]]))


def test_inverse_of_inverse():
    g = Graph('multigraph')
    g.set_auto_simplify(False)
    inp = g.add_vertex(VertexType.BOUNDARY, 0, 0)
    tip1 = g.add_vertex(VertexType.TRIANGLE_INVERSE_INPUT, 0, 1)
    body1 = g.add_vertex(VertexType.TRIANGLE_INVERSE_OUTPUT, 0, 2)
    tip2 = g.add_vertex(VertexType.TRIANGLE_INVERSE_INPUT, 0, 3)
    body2 = g.add_vertex(VertexType.TRIANGLE_INVERSE_OUTPUT, 0, 4)
    outp = g.add_vertex(VertexType.BOUNDARY, 0, 5)
    g.add_edge((inp, tip1), EdgeType.SIMPLE)
    g.add_edge((tip1, body1), EdgeType.W_IO)
    g.add_edge((body1, tip2), EdgeType.SIMPLE)
    g.add_edge((tip2, body2), EdgeType.W_IO)
    g.add_edge((body2, outp), EdgeType.SIMPLE)
    g.set_inputs((inp,))
    g.set_outputs((outp,))
    assert np.allclose(g.to_matrix(), np.array([[1, -2], [0, 1]]))


def test_triangle_times_inverse_is_identity():
    """triangle ; inverse == I  (M · M⁻¹ = I)."""
    g = Graph('multigraph')
    g.set_auto_simplify(False)
    inp = g.add_vertex(VertexType.BOUNDARY, 0, 0)
    tip1 = g.add_vertex(VertexType.TRIANGLE_INPUT, 0, 1)
    body1 = g.add_vertex(VertexType.TRIANGLE_OUTPUT, 0, 2)
    tip2 = g.add_vertex(VertexType.TRIANGLE_INVERSE_INPUT, 0, 3)
    body2 = g.add_vertex(VertexType.TRIANGLE_INVERSE_OUTPUT, 0, 4)
    outp = g.add_vertex(VertexType.BOUNDARY, 0, 5)
    g.add_edge((inp, tip1), EdgeType.SIMPLE)
    g.add_edge((tip1, body1), EdgeType.W_IO)
    g.add_edge((body1, tip2), EdgeType.SIMPLE)
    g.add_edge((tip2, body2), EdgeType.W_IO)
    g.add_edge((body2, outp), EdgeType.SIMPLE)
    g.set_inputs((inp,))
    g.set_outputs((outp,))
    assert np.allclose(g.to_matrix(), np.array([[1, 0], [0, 1]]))


def test_inverse_json_roundtrip():
    g, tip, body = _inverse_graph()
    g2 = Graph.from_json(g.to_json())
    tys = sorted(int(g2.type(v)) for v in g2.vertices())
    assert tys == [int(VertexType.TRIANGLE_INVERSE_INPUT),
                   int(VertexType.TRIANGLE_INVERSE_OUTPUT)]
    assert any(g2.edge_type(e) == EdgeType.W_IO for e in g2.edges())


def test_inverse_tikz_roundtrip():
    g, tip, body = _inverse_graph()
    g.set_inputs(())
    g.set_outputs(())
    s = zxtikz.to_tikz(g)
    g2 = zxtikz.tikz_to_graph(s, backend='multigraph')
    tys = sorted(int(g2.type(v)) for v in g2.vertices())
    assert tys == [int(VertexType.TRIANGLE_INVERSE_INPUT),
                   int(VertexType.TRIANGLE_INVERSE_OUTPUT)]
    assert any(g2.edge_type(e) == EdgeType.W_IO for e in g2.edges())
