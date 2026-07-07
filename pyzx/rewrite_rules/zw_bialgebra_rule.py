# PyZX - Python library for quantum circuit rewriting
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Z-W bialgebra rule.

This rule is the ZW-calculus analogue of the Z-X strong-complementarity
(bialgebra) rule. Its redex is directional: ``inputs -> Z -> W -> outputs``,
i.e. a phase-0 Z-spider whose single downstream leg attaches to a W node's
*input* leg (``W_INPUT``), with the W node's *output* legs (``W_OUTPUT``)
facing the outputs. It rewrites to the complete-bipartite form
``inputs -> W...W -> Z...Z -> outputs``: one W node per input leg, one phase-0
Z-spider per output leg, W-outputs fully connected to the Z-spiders. The
identity is exact (unit scalar), so no scalar correction is applied.
"""

from typing import Dict, List, Optional, Set, Tuple

from pyzx.utils import EdgeType, VertexType, get_w_partner
from pyzx.graph.base import BaseGraph, VT, ET, upair

__all__ = [
    'check_zw_bialgebra',
    'unsafe_zw_bialgebra',
    'zw_bialgebra',
    'match_zw_bialgebra_op',
    'unsafe_zw_bialgebra_op',
    'safe_apply_zw_bialgebra_op',
    'simp_zw_bialgebra_op',
    'is_zw_bialg_op_match',
]

EdgeTab = Dict[Tuple[VT, VT], List[int]]


def _orient_zw(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> Optional[Tuple[VT, VT]]:
    """Return ``(z, w_in)`` if the pair is a Z-spider / W_INPUT pair, else None."""
    for z, w in ((v1, v2), (v2, v1)):
        if g.type(z) == VertexType.Z and g.type(w) == VertexType.W_INPUT:
            return z, w
    return None


def _external_legs(g: BaseGraph[VT, ET], v: VT, exclude: VT) -> List[Tuple[VT, EdgeType]]:
    """Non-W_IO legs of ``v``, as ``(neighbour, edge_type)``, skipping edges to
    ``exclude``."""
    legs: List[Tuple[VT, EdgeType]] = []
    for e in g.incident_edges(v):
        if g.edge_type(e) == EdgeType.W_IO:
            continue
        s, t = g.edge_st(e)
        other = s if s != v else t
        if other == exclude:
            continue
        legs.append((other, g.edge_type(e)))
    return legs


def _bump(etab: EdgeTab, a: VT, b: VT, et: EdgeType) -> None:
    key = upair(a, b)
    if key not in etab:
        etab[key] = [0, 0]
    etab[key][0 if et == EdgeType.SIMPLE else 1] += 1


def check_zw_bialgebra(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    """True iff ``{v1, v2}`` is a phase-0 Z-spider joined to a W node's input leg
    by a single SIMPLE edge, with that Z the W_INPUT's only external neighbour."""
    if not (v1 in g.vertices() and v2 in g.vertices()):
        return False
    pair = _orient_zw(g, v1, v2)
    if pair is None:
        return False
    z, w_in = pair
    if g.phase(z) != 0:
        return False
    if g.num_edges(z, z) != 0 or g.num_edges(w_in, w_in) != 0:
        return False
    edges = list(g.edges(z, w_in))
    if len(edges) != 1 or g.edge_type(edges[0]) != EdgeType.SIMPLE:
        return False
    w_out = get_w_partner(g, w_in)
    if g.num_edges(w_out, w_out) != 0:
        return False
    # W_INPUT must have no external neighbour other than z.
    if _external_legs(g, w_in, exclude=z):
        return False
    # The Z must carry at least one input leg: with none, the expand is not a
    # tensor identity for m >= 2 (the outputs become uncorrelated).
    if len(_external_legs(g, z, exclude=w_in)) == 0:
        return False
    return True


def unsafe_zw_bialgebra(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    """Expand a checked Z / W_INPUT pair into the bipartite W/Z form."""
    pair = _orient_zw(g, v1, v2)
    assert pair is not None
    z, w_in = pair
    w_out = get_w_partner(g, w_in)

    inputs = _external_legs(g, z, exclude=w_in)     # (neighbour, edge_type)
    outputs = _external_legs(g, w_out, exclude=w_in)

    zq, zr = g.qubit(z), g.row(z)
    wq, wr = g.qubit(w_out), g.row(w_out)

    g.remove_vertices([z, w_in, w_out])

    etab: EdgeTab = {}
    new_w_outs: List[VT] = []
    for nbr, et in inputs:
        q = 0.5 * (g.qubit(nbr) + zq)
        r = 0.5 * (g.row(nbr) + zr)
        nw_in = g.add_vertex(VertexType.W_INPUT, q, r)
        nw_out = g.add_vertex(VertexType.W_OUTPUT, q, r + 0.3)
        g.add_edge((nw_in, nw_out), EdgeType.W_IO)
        _bump(etab, nbr, nw_in, et)
        new_w_outs.append(nw_out)

    new_zs: List[VT] = []
    for nbr, et in outputs:
        q = 0.5 * (g.qubit(nbr) + wq)
        r = 0.5 * (g.row(nbr) + wr)
        nz = g.add_vertex(VertexType.Z, q, r)
        g.set_phase(nz, 0)
        _bump(etab, nbr, nz, et)
        new_zs.append(nz)

    for nw_out in new_w_outs:
        for nz in new_zs:
            _bump(etab, nw_out, nz, EdgeType.SIMPLE)

    g.add_edge_table(etab)
    return True


def zw_bialgebra(g: BaseGraph[VT, ET], v1: VT, v2: VT) -> bool:
    if not check_zw_bialgebra(g, v1, v2):
        return False
    return unsafe_zw_bialgebra(g, v1, v2)


def match_zw_bialgebra_op(
    g: BaseGraph[VT, ET],
    vertices: Optional[List[VT]] = None,
) -> Optional[Tuple[List[Tuple[VT, VT]], List[VT]]]:
    """Match a complete-bipartite W/Z region. Returns ``(w_pairs, z_spiders)``
    with ``w_pairs`` a list of ``(w_in, w_out)`` tuples, or None."""
    candidates = set(vertices) if vertices is not None else g.vertex_set()

    w_pairs: List[Tuple[VT, VT]] = []
    seen: Set[VT] = set()
    for v in candidates:
        if g.type(v) == VertexType.W_INPUT and v not in seen:
            w_out = get_w_partner(g, v)
            if w_out not in candidates:
                return None
            w_pairs.append((v, w_out))
            seen.add(v)
            seen.add(w_out)
    z_spiders = [v for v in candidates if g.type(v) == VertexType.Z]

    if len(w_pairs) <= 1 or len(z_spiders) <= 1:
        return None

    z_set = set(z_spiders)
    w_out_set = {w_out for _, w_out in w_pairs}
    region = {v for pair in w_pairs for v in pair} | z_set

    for z in z_spiders:
        if g.phase(z) != 0 or g.num_edges(z, z) != 0:
            return None

    # Each W node: W_INPUT has exactly one external leg; W_OUTPUT connects to
    # every Z spider by a single SIMPLE edge and to nothing else external.
    for w_in, w_out in w_pairs:
        if g.num_edges(w_in, w_in) != 0 or g.num_edges(w_out, w_out) != 0:
            return None
        w_in_ext = _external_legs(g, w_in, exclude=w_in)
        if len(w_in_ext) != 1 or w_in_ext[0][0] in region:
            return None
        out_legs = _external_legs(g, w_out, exclude=w_in)
        if len(out_legs) != len(z_spiders):
            return None
        if {n for n, _ in out_legs} != z_set:
            return None
        if any(et != EdgeType.SIMPLE for _, et in out_legs):
            return None

    # Each Z spider: connects to every W_OUTPUT by a single SIMPLE edge, plus
    # exactly one external (non-W) output leg.
    for z in z_spiders:
        legs = _external_legs(g, z, exclude=z)
        w_conns = [(o, et) for o, et in legs if o in w_out_set]
        ext = [(o, et) for o, et in legs if o not in w_out_set]
        if len(w_conns) != len(w_pairs) or {o for o, _ in w_conns} != w_out_set:
            return None
        if any(et != EdgeType.SIMPLE for _, et in w_conns):
            return None
        if len(ext) != 1 or ext[0][0] in region:
            return None

    return w_pairs, z_spiders


def unsafe_zw_bialgebra_op(
    g: BaseGraph[VT, ET],
    matches: Tuple[List[Tuple[VT, VT]], List[VT]],
) -> bool:
    """Collapse a matched bipartite W/Z region into a single Z -> W pair."""
    w_pairs, z_spiders = matches
    w_out_set = {w_out for _, w_out in w_pairs}

    inputs: List[Tuple[VT, EdgeType]] = []   # from the W nodes' input legs
    for w_in, _w_out in w_pairs:
        inputs.extend(_external_legs(g, w_in, exclude=w_in))

    outputs: List[Tuple[VT, EdgeType]] = []  # from the Z spiders' output legs
    for z in z_spiders:
        for nbr, et in _external_legs(g, z, exclude=z):
            if nbr not in w_out_set:
                outputs.append((nbr, et))

    zq = sum(g.qubit(w_in) for w_in, _ in w_pairs) / len(w_pairs)
    zr = sum(g.row(w_in) for w_in, _ in w_pairs) / len(w_pairs)
    wq = sum(g.qubit(z) for z in z_spiders) / len(z_spiders)
    wr = sum(g.row(z) for z in z_spiders) / len(z_spiders)

    g.remove_vertices([v for pair in w_pairs for v in pair] + z_spiders)

    new_z = g.add_vertex(VertexType.Z, zq, zr)
    g.set_phase(new_z, 0)
    nw_in = g.add_vertex(VertexType.W_INPUT, wq, wr)
    nw_out = g.add_vertex(VertexType.W_OUTPUT, wq, wr + 0.3)
    g.add_edge((nw_in, nw_out), EdgeType.W_IO)
    g.add_edge((new_z, nw_in), EdgeType.SIMPLE)

    etab: EdgeTab = {}
    for nbr, et in inputs:
        _bump(etab, nbr, new_z, et)
    for nbr, et in outputs:
        _bump(etab, nbr, nw_out, et)
    g.add_edge_table(etab)
    return True


def safe_apply_zw_bialgebra_op(g: BaseGraph[VT, ET], vertices: List[VT]) -> bool:
    checked = [v for v in g.vertices() if v in set(vertices)]
    matches = match_zw_bialgebra_op(g, checked)
    if matches is None:
        return False
    return unsafe_zw_bialgebra_op(g, matches)


def simp_zw_bialgebra_op(g: BaseGraph[VT, ET]) -> bool:
    matches = match_zw_bialgebra_op(g)
    if matches is None:
        return False
    return unsafe_zw_bialgebra_op(g, matches)


def is_zw_bialg_op_match(g: BaseGraph[VT, ET], vertices: List[VT]) -> bool:
    return match_zw_bialgebra_op(g, vertices) is not None
