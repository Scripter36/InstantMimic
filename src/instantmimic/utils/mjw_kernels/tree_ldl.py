"""Tree-structured L'DL factor+solve of inertia-like matrices, generated per model.

MuJoCo stores M (and the implicit qH = M - h qDeriv, which keeps M's sparsity)
in CSR over each dof's ancestors: row k lists its ancestors, root first, and
ends with the diagonal, and every ancestor's row is a prefix of row k. Factoring
from the leaves up then creates no fill-in: dof k updates only entries of its
ancestors' rows that are ancestors of both. That schedule depends on the model
alone, so it is emitted as straight-line CUDA C++ with constant indices.

The tree is cut into a serial trunk (the dofs nearest the roots) and parts (the
subtrees hanging off it), packed into at most ``MAX_GROUPS`` groups by
longest-processing-time. A CUDA block serves 32 worlds with one warp per group:
all lanes of a warp run the same code for their own worlds, so nothing
diverges, and each thread keeps only its group's rows in registers. A part
writes its own rows and accumulates into its trunk ancestors' rows; those
contributions pass through shared memory and one warp sums them in a fixed
order before finishing the trunk, so results are deterministic. On the CPU the
same plan runs serially, one thread per world.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import warp as wp

MAX_GROUPS = 4
# Static shared memory available to one block.
MAX_SHARED_BYTES = 48 * 1024
# The tree solve replaces the dense tile Cholesky when the plan's critical path
# (``span``, update units run by one thread) is short, or short relative to the
# dense flops per world (sum over trees of n^3 / 3). A long single-thread path
# spills registers. Measured at 4096 worlds against the tile kernel:
# - spans up to 3.6k always won: ant 2.0x, panda 1.9x, Bolt 4.0x, SNU 10.9x and
#   a 20-hinge chain (span 3.6k) 1.7x;
# - longer chains lost: 26-44 hinges (span 6.5k-23k, dense/span 1.7) 0.5-0.9x,
#   the ostrich (span 14k, dense/span 4.1) 1.0x;
# - a long span still won against a much larger dense cost: SMPL (5.1k, 13.5)
#   4.9x.
MAX_SHORT_SPAN = 4096
MIN_DENSE_TO_SPAN_RATIO = 8.0


@dataclass(frozen=True)
class TreeLdlPlan:
    groups: tuple[tuple[int, ...], ...]
    """Dofs of each group's parts, leaves first (descending)."""
    trunk: tuple[int, ...]
    """Trunk dofs, ascending; ancestor-closed."""
    span: int
    """Critical path in update units: the trunk plus the heaviest group."""
    dense_cost: int
    """Dense Cholesky flops per world over all trees."""


@dataclass(frozen=True)
class TreeFactorSolveScratch:
    kernel: Any
    groups: int
    cuda: bool


def tree_ldl_structure(
    rowadr: np.ndarray, rownnz: np.ndarray, colind: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """The CSR as tuples when it has the nested-ancestor form; else ``None``."""
    rowadr = np.asarray(rowadr, dtype=np.int64)
    rownnz = np.asarray(rownnz, dtype=np.int64)
    colind = np.asarray(colind, dtype=np.int64)
    nv = rowadr.size
    if nv == 0 or rownnz.size != nv or np.any(rownnz < 1):
        return None
    if np.any(rowadr + rownnz > colind.size):
        return None
    for k in range(nv):
        row = colind[rowadr[k] : rowadr[k] + rownnz[k]]
        if row[-1] != k or np.any(row[:-1] >= k):
            return None
        for p, i in enumerate(row[:-1]):
            if rownnz[i] != p + 1 or not np.array_equal(
                colind[rowadr[i] : rowadr[i] + rownnz[i]], row[: p + 1]
            ):
                return None
    return (
        tuple(int(v) for v in rowadr),
        tuple(int(v) for v in rownnz),
        tuple(int(v) for v in colind),
    )


def _trunk_entries(rowadr, rownnz, trunk) -> list[int]:
    return [e for i in trunk for e in range(rowadr[i], rowadr[i] + rownnz[i])]


def _shared_bytes(rowadr, rownnz, groups: int, trunk) -> int:
    if not trunk:
        return 0
    entries = len(_trunk_entries(rowadr, rownnz, trunk))
    return 4 * 32 * (groups * (entries + len(trunk)) + len(trunk))


def plan_tree_ldl(
    rowadr: tuple[int, ...],
    rownnz: tuple[int, ...],
    colind: tuple[int, ...],
    groups: int,
) -> TreeLdlPlan:
    """The plan with the shortest critical path for exactly ``groups`` groups.

    Starting from whole trees, the heaviest splittable part repeatedly moves
    its root dof into the trunk and its children become parts; the cut with
    the shortest estimated critical path is kept.
    """
    nv = len(rowadr)
    parent = [
        colind[rowadr[k] + rownnz[k] - 2] if rownnz[k] > 1 else -1 for k in range(nv)
    ]
    children: list[list[int]] = [[] for _ in range(nv)]
    for k, p in enumerate(parent):
        if p >= 0:
            children[p].append(k)
    # One elimination is n (n - 1) / 2 updates plus the row scaling and solve.
    cost = [n * (n - 1) // 2 + 2 * n for n in rownnz]
    subtree = list(cost)
    size = [1] * nv
    for k in range(nv - 1, -1, -1):
        if parent[k] >= 0:
            subtree[parent[k]] += subtree[k]
            size[parent[k]] += size[k]
    roots = [k for k in range(nv) if parent[k] < 0]
    dense_cost = sum(size[r] ** 3 // 3 for r in roots)

    def pack(parts):
        loads = [0] * groups
        members: list[list[int]] = [[] for _ in range(groups)]
        for r in sorted(parts, key=lambda r: (-subtree[r], r)):
            g = loads.index(min(loads))
            loads[g] += subtree[r]
            members[g].append(r)
        return max(loads), members

    parts, trunk = list(roots), []
    best = None
    while True:
        heaviest, members = pack(parts)
        span = heaviest + sum(cost[k] + 2 * rownnz[k] for k in trunk)
        if best is None or span < best[0]:
            best = (span, sorted(trunk), members)
        splittable = [r for r in parts if children[r]]
        if not splittable:
            break
        r = max(splittable, key=lambda r: (subtree[r], -r))
        parts.remove(r)
        trunk.append(r)
        parts += children[r]
    span, trunk, members = best
    group_nodes = []
    for part_roots in members:
        nodes, stack = [], list(part_roots)
        while stack:
            k = stack.pop()
            nodes.append(k)
            stack += children[k]
        group_nodes.append(tuple(sorted(nodes, reverse=True)))
    return TreeLdlPlan(
        groups=tuple(group_nodes), trunk=tuple(trunk), span=span, dense_cost=dense_cost
    )


def best_tree_ldl_plan(
    rowadr: tuple[int, ...], rownnz: tuple[int, ...], colind: tuple[int, ...]
) -> TreeLdlPlan:
    """The fastest plan whose trunk exchange fits in shared memory (fewest groups on ties)."""
    best = None
    for groups in range(1, MAX_GROUPS + 1):
        plan = plan_tree_ldl(rowadr, rownnz, colind, groups)
        if _shared_bytes(rowadr, rownnz, groups, plan.trunk) > MAX_SHARED_BYTES:
            continue
        if best is None or plan.span < best.span:
            best = plan
    assert best is not None  # one group has no trunk exchange
    return best


def tree_ldl_beats_dense(plan: TreeLdlPlan) -> bool:
    """Whether the plan is expected to beat the dense tile Cholesky (see above)."""
    return (
        plan.span <= MAX_SHORT_SPAN
        or plan.dense_cost >= MIN_DENSE_TO_SPAN_RATIO * plan.span
    )


def _generate_source(
    rowadr: tuple[int, ...],
    rownnz: tuple[int, ...],
    colind: tuple[int, ...],
    plan: TreeLdlPlan,
    cuda: bool,
) -> str:
    groups = len(plan.groups)
    trunk = set(plan.trunk)
    entries = _trunk_entries(rowadr, rownnz, plan.trunk)
    slot = {e: n for n, e in enumerate(entries)}
    dof_slot = {i: n for n, i in enumerate(plan.trunk)}
    T, Tv = max(len(entries), 1), max(len(plan.trunk), 1)
    store = "if (live) xr" if cuda else "xr"
    barrier = f'asm volatile("bar.sync 1, {32 * groups};" ::: "memory");'

    def row(k):
        return range(rowadr[k], rowadr[k] + rownnz[k])

    def eliminate(k, m, d, s, target):
        """Leaves-first elimination of dof k plus its share of the L^-T y solve.

        ``m``, ``d`` and ``s`` prefix k's row, inverse pivot and solution;
        ``target(anc, index, kind)`` names an ancestor's matrix or solution entry.
        """
        start, count = rowadr[k], rownnz[k] - 1
        lines = [f"const float {d}{k} = 1.0f / {m}{start + count};"]
        for p in range(count):
            anc = colind[start + p]
            for q in range(p + 1):
                t = target(anc, rowadr[anc] + q, "m")
                lines.append(f"{t} = {t} - {m}{start + p} * {m}{start + q} * {d}{k};")
        lines += [f"{m}{start + p} = {m}{start + p} * {d}{k};" for p in range(count)]
        for p in range(count):
            t = target(colind[start + p], colind[start + p], "s")
            lines.append(f"{t} = {t} - {m}{start + p} * {s}{k};")
        return lines

    def part_forward(j):
        nodes = plan.groups[j]
        touched_m = {
            rowadr[colind[rowadr[k] + p]] + q
            for k in nodes
            for p in range(rownnz[k] - 1)
            if colind[rowadr[k] + p] in trunk
            for q in range(p + 1)
        }
        touched_s = {
            colind[rowadr[k] + p] for k in nodes for p in range(rownnz[k] - 1)
        } & trunk

        def target(anc, index, kind):
            if anc not in trunk:
                return f"{kind}{index}"
            return f"a{j}_{index}" if kind == "m" else f"b{j}_{index}"

        lines = [f"float m{e} = Mr[{e}];" for k in nodes for e in row(k)]
        lines += [f"float s{k} = yr[{k}];" for k in nodes]
        lines += [f"float a{j}_{e} = 0.0f;" for e in sorted(touched_m)]
        lines += [f"float b{j}_{i} = 0.0f;" for i in sorted(touched_s)]
        for k in nodes:
            lines += eliminate(k, "m", "d", "s", target)
        lines += [f"dm[{j}][{slot[e]}][lw] = a{j}_{e};" for e in sorted(touched_m)]
        lines += [f"ds[{j}][{dof_slot[i]}][lw] = b{j}_{i};" for i in sorted(touched_s)]
        return lines, touched_m, touched_s

    def part_backward(j):
        lines, own = [], set()
        for i in reversed(plan.groups[j]):
            lines.append(f"float x{i} = s{i} * d{i};")
            for p in range(rownnz[i] - 1):
                anc = colind[rowadr[i] + p]
                xa = f"x{anc}" if anc in own else f"tx[{dof_slot[anc]}][lw]"
                lines.append(f"x{i} = x{i} - m{rowadr[i] + p} * {xa};")
            own.add(i)
            lines.append(f"{store}[{i}] = x{i};")
        return lines

    def trunk_solve(touched):
        lines = []
        for e in entries:
            terms = "".join(
                f" + dm[{j}][{slot[e]}][lw]"
                for j in range(groups)
                if e in touched[j][0]
            )
            lines.append(f"float t{e} = Mr[{e}]{terms};")
        for i in plan.trunk:
            terms = "".join(
                f" + ds[{j}][{dof_slot[i]}][lw]"
                for j in range(groups)
                if i in touched[j][1]
            )
            lines.append(f"float u{i} = yr[{i}]{terms};")
        for k in reversed(plan.trunk):
            lines += eliminate(
                k,
                "t",
                "e",
                "u",
                lambda anc, index, kind: f"{'t' if kind == 'm' else 'u'}{index}",
            )
        for i in plan.trunk:
            lines.append(f"u{i} = u{i} * e{i};")
            for p in range(rownnz[i] - 1):
                lines.append(
                    f"u{i} = u{i} - t{rowadr[i] + p} * u{colind[rowadr[i] + p]};"
                )
            lines += [f"tx[{dof_slot[i]}][lw] = u{i};", f"{store}[{i}] = u{i};"]
        return lines

    forwards = [part_forward(j) for j in range(groups)]
    touched = [(m, s) for _, m, s in forwards]
    owner = min(
        range(groups), key=lambda j: sum(rownnz[k] ** 2 for k in plan.groups[j])
    )
    if cuda:
        lines = [
            "const int lw = w & 31;",
            f"const int part = (w >> 5) % {groups};",
            f"const int world = (w / {32 * groups}) * 32 + lw;",
            "const bool live = world < nworld;",
            "const int src = live ? world : nworld - 1;",
        ]
        if trunk:
            lines += [
                f"__shared__ float dm[{groups}][{T}][32];",
                f"__shared__ float ds[{groups}][{Tv}][32];",
                f"__shared__ float tx[{Tv}][32];",
            ]
        lines += [
            "const float* Mr = wp::address(M, src, 0);",
            "const float* yr = wp::address(y, src, 0);",
            "float* xr = wp::address(x, src, 0);",
            "switch (part) {",
        ]
        for j in range(groups):
            lines.append(f"case {j}: {{")
            lines += forwards[j][0]
            if trunk:
                lines.append(barrier)
                if j == owner:
                    lines += ["{", *trunk_solve(touched), "}"]
                lines.append(barrier)
            lines += part_backward(j)
            lines.append("} break;")
        lines.append("}")
    else:
        lines = [
            "const int lw = 0;",
            f"float dm[{groups}][{T}][1];",
            f"float ds[{groups}][{Tv}][1];",
            f"float tx[{Tv}][1];",
            "const float* Mr = wp::address(M, w, 0);",
            "const float* yr = wp::address(y, w, 0);",
            "float* xr = wp::address(x, w, 0);",
        ]
        for forward in forwards:
            lines += forward[0]
        if trunk:
            lines += ["{", *trunk_solve(touched), "}"]
        for j in range(groups):
            lines += part_backward(j)
    return "\n".join(lines) + "\n"


@cache
def make_tree_ldl_factor_solve_kernel(
    rowadr: tuple[int, ...],
    rownnz: tuple[int, ...],
    colind: tuple[int, ...],
    plan: TreeLdlPlan,
    cuda: bool,
) -> Any:
    """Kernel solving (L'DL) x = y for every world under ``plan``."""
    snippet = _generate_source(rowadr, rownnz, colind, plan, cuda)

    @wp.func_native(snippet=snippet)
    def tree_ldl_native(
        M: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
        x: wp.array2d(dtype=float),
        w: int,
        nworld: int,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def tree_ldl_factor_solve(
        M: wp.array2d(dtype=float),
        y: wp.array2d(dtype=float),
        x: wp.array2d(dtype=float),
    ):
        tree_ldl_native(M, y, x, wp.tid(), M.shape[0])

    return tree_ldl_factor_solve


def scratch_for_plan(
    structure: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    plan: TreeLdlPlan,
    device: Any,
) -> TreeFactorSolveScratch:
    cuda = wp.get_device(device).is_cuda
    kernel = make_tree_ldl_factor_solve_kernel(*structure, plan, cuda)
    return TreeFactorSolveScratch(kernel=kernel, groups=len(plan.groups), cuda=cuda)


def create_tree_factor_solve_scratch(m: Any) -> TreeFactorSolveScratch | None:
    """The generated factor+solve for this model, or ``None`` when ineligible."""
    if getattr(m, "nv", 0) <= 0:
        return None
    arrays = [getattr(m, name, None) for name in ("M_rowadr", "M_rownnz", "M_colind")]
    if any(array is None for array in arrays):
        return None
    structure = tree_ldl_structure(*(array.numpy() for array in arrays))
    if structure is None:
        return None
    plan = best_tree_ldl_plan(*structure)
    if not tree_ldl_beats_dense(plan):
        return None
    return scratch_for_plan(structure, plan, arrays[0].device)


def tree_factor_solve(
    scratch: TreeFactorSolveScratch, M: wp.array, x: wp.array, y: wp.array
) -> None:
    """x = M^-1 y for an M in the model's CSR layout."""
    # The generated code indexes each world's row as a contiguous float run.
    for array in (M, x, y):
        if array.strides[1] != 4:
            raise ValueError("tree_factor_solve needs rows of contiguous floats")
    nworld = M.shape[0]
    if scratch.cuda:
        dim = -(-nworld // 32) * 32 * scratch.groups
        block_dim = 32 * scratch.groups
    else:
        dim, block_dim = nworld, 1
    wp.launch(scratch.kernel, dim=dim, inputs=[M, y], outputs=[x], block_dim=block_dim)
