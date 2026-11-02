"""Static-pattern sparse L'DL of the Newton Hessian for the fused Newton.

H = M + J'DJ has a sparsity fixed by the model: M couples a dof with its tree
ancestors, and a constraint row couples the dofs of the ancestor chains of the
bodies it touches. The pattern is the union over everything that can emit a row:
native collision candidates and equality constraints from the model, and the body
groups that custom contact producers declare when they are installed. The symbolic
factor is computed once per model; the numeric factor runs left-looking over
elimination-tree levels in the world's block:
- entries hold N = L D (so L[i,k] L[j,k] D[k] = N[i,k] N[j,k] / D[k]),
- slots are row-major, so an entry's terms are runs of contiguous slots,
- each level is one parallel sweep; a level with few long entries splits each entry
  over a lane team and reduces with a fixed shuffle tree,
- the right-hand side rides as an augmented last row (forward solve), the backward
  solve runs the levels in reverse.
Every entry sums its terms in a fixed order, so results are deterministic.

In the fused loop the factor stays resident in the world's shared memory across
iterations; packed H lives in the world's ``hfactor`` rows and receives the changed
rows' updates there (same pair loop and rounding as the dense update). The initial
solve assembles H on the pattern directly. A row or update outside the declared
pattern is dropped from the factor, increments a per-world counter in ``hfactor``
(``violation_counter``) and prints a warning the first time: it means a row producer
coupled bodies it did not declare. The counter accumulates across solves until its
reader resets it, so a periodic check sees every violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Sequence

import numpy as np
import warp as wp


@dataclass(frozen=True)
class SparseHessianPlan:
    nv: int
    nv_pad: int
    block: int
    nslots: int
    nterms: int
    levels: int
    density: float
    path: str  # "fused" (persistent Newton) or "nonfused" (upstream iteration)
    factor_snippet: str
    init_snippet: str
    nonfused_snippet: str


def _chain(body: int, body_parentid, body_dofadr, body_dofnum) -> list[int]:
    out = []
    b = body
    while b > 0:
        out += range(int(body_dofadr[b]), int(body_dofadr[b]) + int(body_dofnum[b]))
        b = int(body_parentid[b])
    return out


def hessian_pattern(m: Any, body_groups: Sequence[Sequence[int]]) -> list[set[int]]:
    """Upper bound of H's off-diagonal pattern (dof adjacency, MuJoCo dof order)."""
    nv = int(m.nv)
    dof_parentid = m.dof_parentid.numpy()
    body_parentid = m.body_parentid.numpy()
    body_dofadr = m.body_dofadr.numpy()
    body_dofnum = m.body_dofnum.numpy()
    adj = [set() for _ in range(nv)]
    for d in range(nv):
        p = int(dof_parentid[d])
        while p >= 0:
            adj[d].add(p)
            adj[p].add(d)
            p = int(dof_parentid[p])

    def clique(bodies):
        dofs = sorted(
            {
                d
                for b in bodies
                for d in _chain(int(b), body_parentid, body_dofadr, body_dofnum)
            }
        )
        for a in dofs:
            adj[a].update(x for x in dofs if x != a)

    for group in body_groups:
        clique(group)
    # equality rows: connect/weld couple two bodies, joint couples two joints' dofs
    if int(m.neq):
        eq_type = m.eq_type.numpy()
        obj1, obj2 = m.eq_obj1id.numpy(), m.eq_obj2id.numpy()
        objtype = m.eq_objtype.numpy()
        site_bodyid = m.site_bodyid.numpy()
        jnt_dofadr = m.jnt_dofadr.numpy()
        for t, a, b, ot in zip(eq_type, obj1, obj2, objtype, strict=True):
            if t in (EQ_CONNECT, EQ_WELD):
                if ot == OBJ_SITE:
                    a, b = site_bodyid[a], site_bodyid[b]
                clique((a, max(int(b), 0)))
            elif t in (EQ_FLEX, EQ_FLEXVERT, EQ_FLEXSTRAIN):
                continue  # inside the flex couplings (elements / interpolation nodes)
            elif t == EQ_JOINT:
                dofs = [int(jnt_dofadr[a])] + ([int(jnt_dofadr[b])] if b >= 0 else [])
                for x in dofs:
                    adj[x].update(y for y in dofs if y != x)
            else:
                raise ValueError(f"equality type {int(t)} is not supported")
    if int(m.nflex):
        _flex_couplings(m, clique)
    # native collision candidates (contact-type filtered, explicit pairs included)
    geom_bodyid = m.geom_bodyid.numpy()
    pairs = m.nxn_geom_pair_filtered
    if pairs.shape[0]:
        for g1, g2 in pairs.numpy():
            clique((geom_bodyid[g1], geom_bodyid[g2]))
    return adj


def _flex_couplings(m: Any, clique) -> None:
    """Bodies one flex row can couple: an element's vertex bodies (elasticity, edge
    equality, element contacts), an interpolated flex's nodes, plus the other side's
    body for flex-geom candidates; colliding flexes are coupled whole."""
    geom_bodyid = m.geom_bodyid.numpy()
    vertadr, vertnum = m.flex_vertadr.numpy(), m.flex_vertnum.numpy()
    vertbody = m.flex_vertbodyid.numpy()
    elemadr, elemnum = m.flex_elemadr.numpy(), m.flex_elemnum.numpy()
    dataadr, dim = m.flex_elemdataadr.numpy(), m.flex_dim.numpy()
    elem = m.flex_elem.numpy()
    interp = m.flex_interp.numpy()
    nodeadr, nodenum = m.flex_nodeadr.numpy(), m.flex_nodenum.numpy()
    nodebody = m.flex_nodebodyid.numpy()
    nflex = int(m.nflex)
    elem_bodies, flex_bodies = {}, []
    for f in range(nflex):
        if interp[f] != 0:
            nodes = [int(b) for b in nodebody[nodeadr[f] : nodeadr[f] + nodenum[f]]]
            flex_bodies.append(nodes)
            clique(nodes)
            for e in range(elemadr[f], elemadr[f] + elemnum[f]):
                elem_bodies[e] = nodes
            continue
        bodies = [int(b) for b in vertbody[vertadr[f] : vertadr[f] + vertnum[f]]]
        flex_bodies.append(bodies)
        k = int(dim[f]) + 1
        for e in range(int(elemnum[f])):
            verts = elem[dataadr[f] + e * k : dataadr[f] + (e + 1) * k]
            eb = [int(vertbody[vertadr[f] + v]) for v in verts]
            elem_bodies[int(elemadr[f]) + e] = eb
            clique(eb)
    # flex-geom contacts (candidates are filtered at run time by contype/conaffinity):
    # one element (or vertex) against the geom's body
    contype, conaff = m.flex_contype.numpy(), m.flex_conaffinity.numpy()
    gtype, gaff = m.geom_contype.numpy(), m.geom_conaffinity.numpy()
    flex_elems = [
        [b for e, b in elem_bodies.items() if elemadr[f] <= e < elemadr[f] + elemnum[f]]
        for f in range(nflex)
    ]
    for g in range(int(m.ngeom)):
        for f in range(nflex):
            if (contype[f] & gaff[g]) or (gtype[g] & conaff[f]):
                for eb in flex_elems[f]:
                    clique([*eb, int(geom_bodyid[g])])
    selfc = m.flex_selfcollide.numpy()
    for f in range(nflex):
        for h in range(f, nflex):
            hits = (contype[f] & conaff[h]) or (contype[h] & conaff[f])
            if hits and (f != h or selfc[f]):
                clique(flex_bodies[f] + flex_bodies[h])


EQ_CONNECT, EQ_WELD, EQ_JOINT = 0, 1, 2
EQ_FLEX, EQ_FLEXVERT, EQ_FLEXSTRAIN = 4, 5, 6
OBJ_SITE = 6
# Largest per-world shared factor (bytes) the generated kernels keep under the 48 KB
# static limit.
MAX_SHARED_BYTES = 46 * 1024
# Above this fill density (factor entries over the dense triangle) the dense blocked
# factor is kept. Measured (RTX 5090, 2048 worlds, training throughput): density
# 0.40-0.45, nv 68/70 models +6-7 %, SNU nv 56 unchanged (its 22 elimination
# levels cost as much as the 4-tile dense factor); denser patterns were not measured.
MAX_DENSITY = 0.6
MIN_NV = 33  # below, the fused Newton uses the exact single-tile Cholesky
# The upstream (non-fused) iteration keeps the dense factor below the fused Newton's
# DOF cap: on an nv 68 model with sparse Jacobians the sparse factor (gathered
# from the dense H, stored between iterations) trained ~10 % slower than the dense one;
# no larger model was measured.
NONFUSED_MIN_NV = 80


def _symbolic(adj_e: list[set[int]]):
    n = len(adj_e)
    upper = [set(a for a in adj_e[k] if a > k) for k in range(n)]
    for k in range(n):
        nb = sorted(upper[k])
        for i, a in enumerate(nb):
            upper[a].update(nb[i + 1 :])
    colstruct = [sorted(upper[k]) for k in range(n)]
    rowstruct = [[] for _ in range(n)]
    for k in range(n):
        for i in colstruct[k]:
            rowstruct[i].append(k)
    parent = [c[0] if c else -1 for c in colstruct]
    level = [0] * n
    for k in range(n):
        if parent[k] >= 0:
            level[parent[k]] = max(level[parent[k]], level[k] + 1)
    levels = [[] for _ in range(max(level) + 1)]
    for k in range(n):
        levels[level[k]].append(k)
    return rowstruct, colstruct, levels


def _team(nentries: int, maxterms: int, lanes: int) -> int:
    best, cost = 1, None
    for T in (1, 2, 4, 8, 16, 32):
        c = -(-nentries // (lanes // T)) * (-(-maxterms // T) + 2 * int(np.log2(T)))
        if cost is None or c < cost:
            best, cost = T, c
    return best


def _arr(name: str, ctype: str, values) -> str:
    vals = ",".join(str(int(v)) for v in values) or "0"
    return f"static const {ctype} {name}[] = {{{vals}}};\n"


def build_plan(
    m: Any, body_groups: Sequence[Sequence[int]], block: int, path: str = "fused"
) -> SparseHessianPlan:
    nv, nv_pad = int(m.nv), int(m.nv_pad)
    adj = hessian_pattern(m, body_groups)
    # eliminate leaves first: elimination index e = nv - 1 - dof
    orig = [nv - 1 - e for e in range(nv)]
    adj_e = [set(nv - 1 - a for a in adj[orig[e]]) for e in range(nv)]
    rowstruct, colstruct, levels = _symbolic(adj_e)
    n = nv
    slot = {}
    for i in range(n):
        for k in rowstruct[i]:
            slot[i, k] = len(slot)
        slot[i, i] = len(slot)
    nfac = len(slot)  # factor slots (rows of L incl. diagonal)
    rhs = nfac
    for k in range(n):
        slot[n, k] = len(slot)
    nslots = len(slot)
    if nslots >= 65536 or nfac + n > nv_pad * nv_pad:
        raise ValueError("sparse Hessian factor does not fit the packed layout")
    rs = [set(r) for r in rowstruct]
    # factor entries per level
    lvl_ptr, lvl_team, ent_slot, ent_col, run_ptr, run_ab, run_kl = (
        [0],
        [],
        [],
        [],
        [0],
        [],
        [],
    )
    nterms = 0
    for cols in levels:
        entries = []
        for j in cols:
            for i in [j, *colstruct[j], n]:
                ks = sorted(rs[j]) if i in (j, n) else sorted(rs[j] & rs[i])
                runs = []
                for k in ks:
                    if (
                        runs
                        and k == runs[-1][2] + runs[-1][3]
                        and slot[i, k] == runs[-1][0] + runs[-1][3]
                        and slot[j, k] == runs[-1][1] + runs[-1][3]
                    ):
                        runs[-1][3] += 1
                    else:
                        runs.append([slot[i, k], slot[j, k], k, 1])
                entries.append((slot[i, j], j if i == j else -1, runs, len(ks)))
        entries.sort(key=lambda e: -e[3])
        for s, c, runs, nt in entries:
            ent_slot.append(s)
            ent_col.append(c)
            for a, b, k, ln in runs:
                run_ab.append(a | (b << 16))
                run_kl.append(k | (ln << 16))
            run_ptr.append(len(run_ab))
            nterms += nt
        lvl_ptr.append(len(ent_slot))
        lvl_team.append(_team(len(entries), max(e[3] for e in entries), block))
    # gather: factor slot (i, j), i >= j in elimination order -> H upper (min, max) in dof order
    gat = []
    for (i, j), s in sorted(slot.items(), key=lambda kv: kv[1]):
        if s >= nfac:
            break
        a, b = orig[i], orig[j]
        gat.append(min(a, b) * nv_pad + max(a, b))
    # forward solve (solve-only path): rows by level; row j's L entries are slots
    # slot(j, k) for k in rowstruct[j] (contiguous), diag last
    f_lvl, f_row = [0], []
    for cols in levels:
        f_row += cols
        f_lvl.append(len(f_row))
    row_start = [
        slot[j, rowstruct[j][0]] if rowstruct[j] else slot[j, j] for j in range(n)
    ]
    rk_ptr, rk = [0], []
    for j in range(n):
        rk += rowstruct[j]
        rk_ptr.append(len(rk))
    # backward solve, root level first: once a level's unknowns are final, each pushes
    # its column terms to its descendants. A column's entries lie on its elimination
    # tree ancestors, whose levels strictly increase, so every unknown receives at most
    # one push per level and a level's pushes are independent.
    b_lvl, b_slot, b_src, b_dst = [0], [], [], []
    for cols in reversed(levels):
        for i in cols:
            for j in rowstruct[i]:
                b_slot.append(slot[i, j])
                b_src.append(i)
                b_dst.append(j)
        b_lvl.append(len(b_slot))
    for a in (
        lvl_ptr,
        ent_slot,
        run_ptr,
        f_lvl,
        f_row,
        row_start,
        rk_ptr,
        rk,
        b_lvl,
        b_slot,
        b_src,
        b_dst,
    ):
        if max(a, default=0) >= 65536:
            raise ValueError("sparse Hessian tables exceed 16-bit indices")
    # incremental update: H upper (row <= col, dof order) -> factor slot, -1 off pattern
    pair = [-1] * (nv * nv)
    for a in range(nv):
        for b in range(a, nv):
            i, j = nv - 1 - a, nv - 1 - b  # elimination indices, i >= j
            if (i, j) in slot and slot[i, j] < nfac:
                pair[a * nv + b] = slot[i, j]
    # initial assembly H = M + J'DJ per slot: its dof pair and M's CSR index (-1 if none)
    csr = {}
    colind = m.M_colind.numpy()
    rowi = m.M_hinit_i.numpy()
    for e, (r, c) in enumerate(zip(rowi, colind, strict=True)):
        csr[min(int(r), int(c)), max(int(r), int(c))] = e
    s_a, s_b, s_m = [], [], []
    for (i, j), sl in sorted(slot.items(), key=lambda kv: kv[1]):
        if sl >= nfac:
            break
        a, b = sorted((orig[i], orig[j]))
        s_a.append(a)
        s_b.append(b)
        s_m.append(csr.get((a, b), -1))
    tables = (
        _arr("SH_LP", "unsigned short", lvl_ptr)
        + _arr("SH_LT", "unsigned char", lvl_team)
        + _arr("SH_ES", "unsigned short", ent_slot)
        + _arr("SH_EC", "short", ent_col)
        + _arr("SH_RP", "unsigned short", run_ptr)
        + _arr("SH_RAB", "unsigned int", run_ab)
        + _arr("SH_RKL", "unsigned int", run_kl)
        + _arr("SH_ORIG", "unsigned short", orig)
        + _arr("SH_BL", "unsigned short", b_lvl)
        + _arr("SH_BS", "unsigned short", b_slot)
        + _arr("SH_BI", "unsigned short", b_src)
        + _arr("SH_BJ", "unsigned short", b_dst)
    )
    fused_tables = (
        _arr("SH_PAIR", "int", pair)
        + _arr("SH_FL", "unsigned short", f_lvl)
        + _arr("SH_FR", "unsigned short", f_row)
        + _arr("SH_RS", "unsigned short", row_start)
        + _arr("SH_RKP", "unsigned short", rk_ptr)
        + _arr("SH_RK", "unsigned short", rk)
    )
    words = (nv + 31) // 32
    pmask = []
    for a in range(nv):
        # the factor's pattern (fill included), as the incremental update's SH_PAIR
        bits = [0] * words
        for b in range(nv):
            if b == a or pair[min(a, b) * nv + max(a, b)] >= 0:
                bits[b // 32] |= 1 << (b % 32)
        pmask += bits
    pmask_table = (
        "static const unsigned int SH_PM[] = {"
        + ",".join(f"{v}u" for v in pmask)
        + "};\n"
    )
    init_tables = (
        pmask_table
        + _arr("SH_SA", "unsigned short", s_a)
        + _arr("SH_SB", "unsigned short", s_b)
        + _arr("SH_SM", "int", s_m)
    )
    from instantmimic.utils.mjw_kernels.private import types

    quad = int(types.ConstraintState.QUADRATIC.value)
    teams = sorted(set(lvl_team))
    nw = block // 32
    nl = len(levels)
    R = 4  # constraint rows per shared tile in the initial assembly

    def body(T):
        shuffles = "".join(
            f"        v += __shfl_down_sync(0xFFFFFFFFu, v, {T >> s}, {T});\n"
            for s in range(1, int(np.log2(T)) + 1)
        )
        return f"""
    if (T == {T}) {{
      const int tl = lane & {T - 1};
      const int team = lane / {T};
      for (int eb = e0; eb < e1; eb += {block // T}) {{
        const int e = eb + team;
        float v = 0.0f;
        int sl = 0, col = -1;
        if (e < e1) {{
          sl = SH_ES[e]; col = SH_EC[e];
          for (int r = SH_RP[e]; r < SH_RP[e + 1]; ++r) {{
            const unsigned int ab = SH_RAB[r], kl = SH_RKL[r];
            const float* pa = N + (ab & 0xFFFFu);
            const float* pb = N + (ab >> 16);
            const float* pd = iD + (kl & 0xFFFFu);
            const int len = (int)(kl >> 16);
            for (int t = tl; t < len; t += {T}) v = fmaf(pa[t] * pb[t], pd[t], v);
          }}
        }}
{shuffles}        if (e < e1 && tl == 0) {{
          const float w = N[sl] - v;
          N[sl] = w;
          if (col >= 0) iD[col] = 1.0f / w;
        }}
      }}
    }}"""

    factor_levels = f"""
  for (int j = lane; j < {n}; j += {block}) N[{rhs} + j] = *wp::address(grad, SH_ORIG[j], 0);
  __syncthreads();
  for (int l = 0; l < {nl}; ++l) {{
    const int e0 = SH_LP[l];
    const int e1 = SH_LP[l + 1];
    const int T = SH_LT[l];
{"".join(body(T) for T in teams)}
    __syncthreads();
  }}
"""
    backward = f"""
for (int l = 0; l < {nl}; ++l) {{
  for (int p = SH_BL[l] + lane; p < SH_BL[l + 1]; p += {block}) {{
    const int i = SH_BI[p];
    const int j = SH_BJ[p];
    N[{rhs} + j] = N[{rhs} + j] - N[SH_BS[p]] * (N[{rhs} + i] * iD[i]);
  }}
  __syncthreads();
}}
float s0 = 0.0f, s1 = 0.0f;
for (int j = lane; j < {n}; j += {block}) {{
  const int d = SH_ORIG[j];
  const float v = N[{rhs} + j] * iD[j];
  *wp::address(search, d, 0) = -v;
  s0 += v * v;
  s1 += *wp::address(grad, d, 0) * v;
}}
for (int o = 16; o > 0; o >>= 1) {{
  s0 += __shfl_down_sync(0xFFFFFFFFu, s0, o);
  s1 += __shfl_down_sync(0xFFFFFFFFu, s1, o);
}}
if ((lane & 31) == 0) {{ red[0][lane >> 5] = s0; red[1][lane >> 5] = s1; }}
__syncthreads();
float ra = 0.0f, rb = 0.0f;
for (int i = 0; i < {nw}; ++i) {{ ra += red[0][i]; rb += red[1][i]; }}
__syncthreads();
return wp::vec2(ra, rb);
"""
    shared = f"""
__shared__ float N[{nslots}];
__shared__ float iD[{n}];
__shared__ float red[2][{nw}];
float* Uf = wp::address(U, 0, 0);
const int lane = lane_in;
"""
    # Fused-loop function. hfactor layout per world: [0, nfac) packed H, [nfac, 2 nfac)
    # the factor N, [2 nfac, 2 nfac + n) 1/D, last element: pattern-violation counter.
    snippet = (
        tables
        + fused_tables
        + shared
        + f"""
if (mode == 2) {{
  for (int s = lane; s < {nfac}; s += {block}) N[s] = Uf[{nfac} + s];
  for (int j = lane; j < {n}; j += {block}) iD[j] = Uf[{2 * nfac} + j];
  __syncthreads();
  return wp::vec2(0.0f, 0.0f);
}}
if (mode == 0) {{
  // incremental H (L2-resident packed copy): pairs of the changed rows' dof union
  const int nu = *wp::address(uni, w, {nv});
  const int npair = nu * (nu + 1) / 2;
  for (int e = lane; e < npair; e += {block}) {{
    const int cj = ((int)sqrtf((float)(1 + 8 * e)) - 1) / 2;
    const int row = *wp::address(uni, w, e - cj * (cj + 1) / 2);
    const int col = *wp::address(uni, w, cj);
    float delta = 0.0f;
    for (int t = 0; t < count; ++t) {{
      const int r = *wp::address(ids, w, t);
      const float a = *wp::address(J, w, r, row);
      const float b = *wp::address(J, w, r, col);
      float da = *wp::address(D, w, r) * a;
      if (*wp::address(state, w, r) != {quad}) da = -da;
      if (a != 0.0f && b != 0.0f) delta = fmaf(da, b, delta);
    }}
    if (delta != 0.0f) {{
      const int sl = SH_PAIR[row * {nv} + col];
      if (sl >= 0) Uf[sl] = Uf[sl] + delta;
      else if (atomicAdd(Uf + {nv_pad * nv_pad - 1}, 1.0f) == 0.0f)
        printf("sparse Hessian: world %d updates H(%d, %d) outside the declared coupling pattern\\n", w, row, col);
    }}
  }}
  __syncthreads();
  for (int s = lane; s < {nfac}; s += {block}) N[s] = Uf[s];
{factor_levels}
}} else {{
  // mode 1: forward solve with the resident factor
  for (int l = 0; l < {nl}; ++l) {{
    for (int q = SH_FL[l] + lane; q < SH_FL[l + 1]; q += {block}) {{
      const int j = SH_FR[q];
      float v = *wp::address(grad, SH_ORIG[j], 0);
      const float* rowp = N + SH_RS[j];
      const int k0 = SH_RKP[j];
      for (int t = 0; t < SH_RKP[j + 1] - k0; ++t) {{
        const int k = SH_RK[k0 + t];
        v = v - rowp[t] * iD[k] * N[{rhs} + k];
      }}
      N[{rhs} + j] = v;
    }}
    __syncthreads();
  }}
}}
"""
        + backward
    )
    # Initial assembly + factor: H = M + sum over QUADRATIC rows of D J' J on the
    # pattern slots, each slot summed in row order. Rows are staged through a small
    # shared tile of {R} rows, so the block's footprint stays close to the factor's
    # (a 16-row tile cost the nv 68/70 models a second partial wave of worlds); the next
    # tile's rows are loaded into registers while the current one is summed.
    pf = -(-R * nv // block)
    prefetch = "".join(
        f"""
    {{
      const int q = lane + {i * block};
      const int r = q / {nv};
      pj[{i}] = (q < {R * nv} && k1 + r < nefc) ? *wp::address(J, w, k1 + r, q - r * {nv}) : 0.0f;
    }}"""
        for i in range(pf)
    )
    stage = "".join(
        f"""
  {{
    const int q = lane + {i * block};
    if (q < {R * nv}) Jc[q / {nv}][q - (q / {nv}) * {nv}] = pj[{i}];
  }}"""
        for i in range(pf)
    )
    init_snippet = (
        tables
        + init_tables
        + shared
        + f"""
__shared__ float Jc[{R}][{nv}];
__shared__ float Dc[{R}];
const float* Mw = wp::address(M, w, 0);
for (int s = lane; s < {nfac}; s += {block}) {{
  const int e = SH_SM[s];
  N[s] = e >= 0 ? Mw[e] : 0.0f;
}}
const int nefc = min(nefc_in, (int)J.shape[1]);
float pj[{pf}];
float pd = 0.0f;
{{
  const int k1 = 0;
  if (k1 < nefc) {{{prefetch}
    if (lane < {R} && k1 + lane < nefc && *wp::address(state, w, k1 + lane) == {quad}) pd = *wp::address(D, w, k1 + lane);
  }}
}}
for (int k = 0; k < nefc; k += {R}) {{
  __syncthreads();{stage}
  if (lane < {R}) Dc[lane] = pd;
  __syncthreads();
  {{
    const int k1 = k + {R};
    pd = 0.0f;
    if (k1 < nefc) {{{prefetch}
      if (lane < {R} && k1 + lane < nefc && *wp::address(state, w, k1 + lane) == {quad}) pd = *wp::address(D, w, k1 + lane);
    }}
  }}
  // every active row's dof support must lie in the pattern (later flips are checked by
  // the incremental update): one warp per row, one dof per lane, masks by ballot
  for (int r = lane >> 5; r < {R}; r += {nw}) {{
    if (Dc[r] == 0.0f) continue;
    const int wl = lane & 31;
    unsigned int rm[{words}];
    for (int q = 0; q < {words}; ++q) {{
      const int d = q * 32 + wl;
      rm[q] = __ballot_sync(0xFFFFFFFFu, d < {nv} && Jc[r][d] != 0.0f);
    }}
    bool bad = false;
    for (int q = 0; q < {words}; ++q) {{
      const int d = q * 32 + wl;
      if (d < {nv} && ((rm[q] >> wl) & 1u))
        for (int p = 0; p < {words}; ++p) bad = bad || (rm[p] & ~SH_PM[d * {words} + p]) != 0u;
    }}
    if (__any_sync(0xFFFFFFFFu, bad) && wl == 0 && atomicAdd(Uf + {nv_pad * nv_pad - 1}, 1.0f) == 0.0f)
      printf("sparse Hessian: world %d has a constraint row outside the declared coupling pattern\\n", w);
  }}
  for (int s = lane; s < {nfac}; s += {block}) {{
    const int a = SH_SA[s];
    const int b = SH_SB[s];
    float v = N[s];
    #pragma unroll
    for (int r = 0; r < {R}; ++r) v = fmaf(Jc[r][a] * Dc[r], Jc[r][b], v);
    N[s] = v;
  }}
}}
__syncthreads();
for (int s = lane; s < {nfac}; s += {block}) Uf[s] = N[s];
{factor_levels}
for (int s = lane; s < {nfac}; s += {block}) Uf[{nfac} + s] = N[s];
for (int j = lane; j < {n}; j += {block}) Uf[{2 * nfac} + j] = iD[j];
"""
        + backward
    )
    # Upstream (non-fused) iteration: factor the dense H the upstream assembly and
    # incremental update maintain; hfactor holds [0, nfac) N, [nfac, nfac + n) 1/D.
    nonfused_snippet = (
        tables
        + _arr("SH_GAT", "unsigned int", gat)
        + "".join(
            t + "\n"
            for t in fused_tables.splitlines()
            if not t.startswith("static const int SH_PAIR")
        )
        + shared
        + f"""
if (mode == 4) {{
  const float* Hp = wp::address(Hd, 0, 0);
  for (int s = lane; s < {nfac}; s += {block}) N[s] = Hp[SH_GAT[s]];
  __syncthreads();
{factor_levels}
  for (int s = lane; s < {nfac}; s += {block}) Uf[s] = N[s];
  for (int j = lane; j < {n}; j += {block}) Uf[{nfac} + j] = iD[j];
}} else {{
  for (int s = lane; s < {nfac}; s += {block}) N[s] = Uf[s];
  for (int j = lane; j < {n}; j += {block}) iD[j] = Uf[{nfac} + j];
  __syncthreads();
  for (int l = 0; l < {nl}; ++l) {{
    for (int q = SH_FL[l] + lane; q < SH_FL[l + 1]; q += {block}) {{
      const int j = SH_FR[q];
      float v = *wp::address(grad, SH_ORIG[j], 0);
      const float* rowp = N + SH_RS[j];
      const int k0 = SH_RKP[j];
      for (int t = 0; t < SH_RKP[j + 1] - k0; ++t) {{
        const int k = SH_RK[k0 + t];
        v = v - rowp[t] * iD[k] * N[{rhs} + k];
      }}
      N[{rhs} + j] = v;
    }}
    __syncthreads();
  }}
}}
"""
        + backward
    )
    if 2 * nfac + n >= nv_pad * nv_pad - 1:
        raise ValueError("packed H and factor do not fit hfactor")
    init_tile = R * nv + R if path == "fused" else 0
    if 4 * (nslots + n + 2 * nw + init_tile) > MAX_SHARED_BYTES:
        raise ValueError("the factor does not fit the block's shared memory")
    if path == "fused":
        nonfused_snippet = ""
    else:
        snippet = init_snippet = ""
    density = nfac / (nv * (nv + 1) / 2)
    return SparseHessianPlan(
        nv,
        nv_pad,
        block,
        nslots,
        nterms,
        nl,
        density,
        path,
        snippet,
        init_snippet,
        nonfused_snippet,
    )


def plan_for_model(
    m: Any,
    ctx: Any,
    body_groups: Sequence[Sequence[int]] | None,
    block: int,
    *,
    compact: bool = False,
    fused_runtime: bool = True,
) -> tuple[SparseHessianPlan | None, str]:
    """The plan when the model qualifies, else ``None`` and the reason.

    The persistent (fused) Newton is preferred; models it does not take (sparse
    Jacobians, more than 79 DOFs) can use the upstream iteration with the sparse
    factor substituted. ``block`` is the fused kernel's block width.
    """
    from instantmimic.utils.mjw_kernels import newton_fusion
    from instantmimic.utils.mjw_kernels.private import solver, types

    if body_groups is None:
        return None, "a custom contact producer did not declare its body groups"
    if not m.body_parentid.device.is_cuda:
        return None, "fused Newton not eligible (CPU)"
    if int(m.nv) < MIN_NV:
        return None, f"nv {int(m.nv)} < {MIN_NV}"
    if int(m.ntendon) and (
        int(m.tendon_limited_adr.shape[0])
        or bool((m.tendon_frictionloss.numpy() > 0).any())
    ):
        return None, "tendon limit or frictionloss rows"
    if fused_runtime and newton_fusion.eligible(m, ctx, compact=compact):
        if int(m.nflex):
            return None, "flex objects on the fused path"
        path, width = "fused", block
    elif (
        m.opt.solver == types.SolverType.NEWTON
        and m.opt.cone == types.ConeType.PYRAMIDAL
        and m.opt.integrator != types.IntegratorType.DISCRETE
        and m.opt.iterations > 0
        and not compact
        and not solver._sparse_compact(ctx)
        and ctx.hfactor.shape[1] == int(m.nv_pad)
    ):
        if int(m.nv) < NONFUSED_MIN_NV:
            return None, f"upstream iteration with nv {int(m.nv)} < {NONFUSED_MIN_NV}"
        path, width = "nonfused", 64 if int(m.nv) <= 96 else 128
    else:
        return None, "Newton solver structure not supported"
    try:
        plan = build_plan(m, body_groups, width, path)
    except ValueError as exc:
        return None, str(exc)
    if plan.density > MAX_DENSITY:
        return None, f"pattern density {plan.density:.2f} > {MAX_DENSITY}"
    return plan, "enabled"


@cache
def native_func(plan: SparseHessianPlan):
    """The fused loop's device function: mode 0 update + factor + solve, 1 solve with
    the resident factor, 2 load the initial factor into the block."""

    @wp.func_native(snippet=plan.factor_snippet)
    def sparse_newton_native(
        J: wp.array3d(dtype=float),
        D: wp.array2d(dtype=float),
        state: wp.array2d(dtype=int),
        ids: wp.array2d(dtype=int),
        count: int,
        uni: wp.array2d(dtype=int),
        w: int,
        Hd: wp.array2d(dtype=float),
        grad: wp.array2d(dtype=float),
        U: wp.array2d(dtype=float),
        search: wp.array2d(dtype=float),
        lane_in: int,
        mode: int,
    ) -> wp.vec2: ...

    return sparse_newton_native


def violation_counter(hfactor: wp.array, plan: SparseHessianPlan) -> wp.array:
    """Fused path: a per-world view of the pattern-violation counter, the last
    element of each world's ``hfactor``. It counts active constraint rows at the
    initial assembly and incremental H updates outside the pattern (must stay zero)
    and accumulates across solves; zero it when the plan is installed and after a
    nonzero read. The upstream-iteration path has no counter (validate with
    ``pattern_misses``)."""
    if plan.path != "fused":
        raise ValueError("the violation counter exists on the fused path only")
    last = plan.nv_pad - 1
    return hfactor[:, last, last]


def violations(hfactor: wp.array, plan: SparseHessianPlan) -> np.ndarray:
    """Per-world violation counts accumulated since the counter was last zeroed."""
    return violation_counter(hfactor, plan).numpy()


@cache
def init_factor_kernel(plan: SparseHessianPlan):
    """Initial H = M + J'DJ on the pattern, factor and solve; fills hfactor."""

    @wp.func_native(snippet=plan.init_snippet)
    def sparse_newton_init_native(
        J: wp.array3d(dtype=float),
        D: wp.array2d(dtype=float),
        state: wp.array2d(dtype=int),
        nefc_in: int,
        M: wp.array2d(dtype=float),
        w: int,
        grad: wp.array2d(dtype=float),
        U: wp.array2d(dtype=float),
        search: wp.array2d(dtype=float),
        lane_in: int,
    ) -> wp.vec2: ...

    @wp.kernel(module="unique", enable_backward=False)
    def sparse_hessian_init_factor(
        done: wp.array(dtype=bool),
        nefc: wp.array(dtype=int),
        J: wp.array3d(dtype=float),
        D: wp.array2d(dtype=float),
        state: wp.array2d(dtype=int),
        M: wp.array2d(dtype=float),
        grad: wp.array3d(dtype=float),
        U: wp.array3d(dtype=float),
        search: wp.array3d(dtype=float),
        search_dot: wp.array(dtype=float),
        decrement: wp.array(dtype=float),
    ):
        w, lane = wp.tid()
        if done[w]:
            return
        sums = sparse_newton_init_native(
            J, D, state, nefc[w], M, w, grad[w], U[w], search[w], lane
        )
        if lane == 0:
            search_dot[w] = sums[0]
            decrement[w] = sums[1]

    return sparse_hessian_init_factor


@cache
def nonfused_kernel(plan: SparseHessianPlan, skip_unchanged: bool, skip_noflip: bool):
    """Upstream ``_cholesky_factorize_solve`` kernels with the sparse factor: refactor
    when rows changed QUADRATIC state (always without ``skip_unchanged``), else solve
    with the stored factor; ``skip_noflip`` leaves worlds without state changes."""

    @wp.func_native(snippet=plan.nonfused_snippet)
    def sparse_newton_upstream_native(
        Hd: wp.array2d(dtype=float),
        grad: wp.array2d(dtype=float),
        U: wp.array2d(dtype=float),
        search: wp.array2d(dtype=float),
        lane_in: int,
        mode: int,
    ) -> wp.vec2: ...

    SKIP_UNCHANGED, SKIP_NOFLIP = skip_unchanged, skip_noflip

    @wp.kernel(module="unique", enable_backward=False)
    def sparse_hessian_upstream_factor(
        done: wp.array(dtype=bool),
        grad: wp.array3d(dtype=float),
        H: wp.array3d(dtype=float),
        quad_changed: wp.array(dtype=int),
        state_changed: wp.array(dtype=int),
        U: wp.array3d(dtype=float),
        search: wp.array3d(dtype=float),
        search_dot: wp.array(dtype=float),
        decrement: wp.array(dtype=float),
    ):
        w, lane = wp.tid()
        if done[w]:
            return
        if wp.static(SKIP_NOFLIP):
            if state_changed[w] == 0:
                return
        mode = int(4)
        if wp.static(SKIP_UNCHANGED):
            if quad_changed[w] == 0:
                mode = 5
        sums = sparse_newton_upstream_native(H[w], grad[w], U[w], search[w], lane, mode)
        if lane == 0:
            search_dot[w] = sums[0]
            decrement[w] = sums[1]

    return sparse_hessian_upstream_factor


def _rebind(fn: Any, overrides: dict[str, Any]) -> Any:
    from types import FunctionType

    raw = getattr(fn, "__wrapped__", fn)
    namespace = dict(raw.__globals__)
    namespace.update(overrides)
    return FunctionType(
        raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__
    )


@cache
def nonfused_solver_functions(plan: SparseHessianPlan):
    """``(update_gradient, solver_iteration)``: the upstream functions with the dense
    Hessian factorization replaced by this plan's sparse factor."""
    from instantmimic.utils.mjw_kernels.private import solver

    def cholesky_factorize_solve(m, d, ctx, skip_unchanged=False, skip_noflip=False):
        state = ctx.state_changed_count if skip_noflip else ctx.quad_changed_count
        wp.launch_tiled(
            nonfused_kernel(plan, bool(skip_unchanged), bool(skip_noflip)),
            dim=d.nworld,
            inputs=[
                ctx.done,
                ctx.grad.reshape((d.nworld, ctx.grad.shape[1], 1)),
                ctx.h,
                ctx.quad_changed_count,
                state,
                ctx.hfactor,
                ctx.search.reshape((d.nworld, m.nv, 1)),
                ctx.search_dot,
                ctx.newton_decrement,
            ],
            block_dim=plan.block,
        )

    override = {"_cholesky_factorize_solve": cholesky_factorize_solve}
    update_gradient = _rebind(solver._update_gradient, override)
    incremental = _rebind(solver._update_gradient_incremental, override)
    iteration = _rebind(
        solver._solver_iteration,
        {
            "_update_gradient": update_gradient,
            "_update_gradient_incremental": incremental,
        },
    )
    return update_gradient, iteration


def pattern_misses(h: np.ndarray, plan_adj: list[set[int]], tol: float = 0.0) -> int:
    """Count nonzero upper-triangle entries of dense Hessians ``h`` (nworld, nv_pad,
    nv_pad) outside a dof adjacency (validation helper for the upstream path)."""
    nv = len(plan_adj)
    mask = np.zeros((nv, nv), bool)
    for a in range(nv):
        mask[a, a] = True
        for b in plan_adj[a]:
            mask[a, b] = True
    upper = np.triu(np.ones((nv, nv), bool))
    off = upper & ~mask
    return int((np.abs(h[:, :nv, :nv][:, off]) > tol).sum())
