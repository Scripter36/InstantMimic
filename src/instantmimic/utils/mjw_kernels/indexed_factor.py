"""Blocked Cholesky with direct inertia indexing and unpadded RHS/output.

Copyright 2025 The Newton Developers, licensed under the Apache License,
Version 2.0. Derived from mujoco_warp block_cholesky.py.
Local changes index the inertia matrix directly instead of gathering through CSR
slot indices, and keep the right-hand side and output unpadded."""

from functools import cache

import warp as wp

from instantmimic.utils.warp_kernels import expose_kernel_name


@cache
def create_indexed_cholesky_factor_solve_func(
    block_size: int,
    matrix_size_static: int,
    vector_size_static: int,
):
    block_area = block_size * block_size
    nblk = matrix_size_static // block_size

    @wp.func
    def load_block(
        M_row: wp.array[float],
        block_idx: wp.array[int],
        kb: int,
        ib: int,
    ):
        idx = wp.tile_load(
            block_idx,
            shape=(block_area,),
            offset=((kb * nblk + ib) * block_area,),
            storage="shared",
        )
        flat = wp.tile_load_indexed(
            M_row,
            idx,
            shape=(block_area,),
            storage="shared",
        )
        return wp.tile_reshape(flat, (block_size, block_size))

    @wp.func
    def func(
        M_row: wp.array[float],
        block_idx: wp.array[int],
        pad_identity: wp.array[float],
        y: wp.array2d[float],
        matrix_size: int,
        nv: int,
        U: wp.array2d[float],
        x: wp.array2d[float],
    ):
        rhs_tile = wp.tile_zeros(
            shape=(matrix_size_static, 1),
            dtype=float,
            storage="shared",
        )
        active_rhs = wp.tile_load(
            y,
            shape=(vector_size_static, 1),
            offset=(0, 0),
            storage="shared",
            bounds_check=False,
        )
        wp.tile_assign(rhs_tile, active_rhs, offset=(0, 0))

        for k in range(0, matrix_size, block_size):
            kb = k // block_size
            end = k + block_size
            rhs_view = wp.tile_view(
                rhs_tile,
                shape=(block_size, 1),
                offset=(k, 0),
            )
            A_kk_tile = load_block(M_row, block_idx, kb, kb)
            if end > nv:
                pad_flat = wp.tile_load(
                    pad_identity,
                    shape=(block_area,),
                    offset=(0,),
                    storage="shared",
                )
                pad_tile = wp.tile_reshape(pad_flat, (block_size, block_size))
                A_kk_tile = wp.tile_map(wp.add, A_kk_tile, pad_tile)

            for j in range(0, k, block_size):
                U_block = wp.tile_load(
                    U,
                    shape=(block_size, block_size),
                    offset=(j, k),
                    storage="shared",
                )
                wp.tile_matmul(
                    wp.tile_transpose(U_block),
                    U_block,
                    A_kk_tile,
                    alpha=-1.0,
                )
                y_block = wp.tile_view(
                    rhs_tile,
                    shape=(block_size, 1),
                    offset=(j, 0),
                )
                wp.tile_matmul(
                    wp.tile_transpose(U_block),
                    y_block,
                    rhs_view,
                    alpha=-1.0,
                )

            U_kk = wp.tile_cholesky(A_kk_tile, fill_mode="upper")
            wp.tile_store(U, U_kk, offset=(k, k))
            wp.tile_lower_solve_inplace(wp.tile_transpose(U_kk), rhs_view)

            for i in range(end, matrix_size, block_size):
                ib = i // block_size
                A_ki_tile = load_block(M_row, block_idx, kb, ib)
                for j in range(0, k, block_size):
                    U_jk_tile = wp.tile_load(
                        U,
                        shape=(block_size, block_size),
                        offset=(j, k),
                        storage="shared",
                    )
                    U_ji_tile = wp.tile_load(
                        U,
                        shape=(block_size, block_size),
                        offset=(j, i),
                        storage="shared",
                    )
                    wp.tile_matmul(
                        wp.tile_transpose(U_jk_tile),
                        U_ji_tile,
                        A_ki_tile,
                        alpha=-1.0,
                    )
                wp.tile_lower_solve_inplace(wp.tile_transpose(U_kk), A_ki_tile)
                wp.tile_store(U, A_ki_tile, offset=(k, i))

        for i in range(matrix_size - block_size, -1, -block_size):
            i_end = i + block_size
            tmp_tile = wp.tile_view(
                rhs_tile,
                shape=(block_size, 1),
                offset=(i, 0),
            )
            for j in range(i_end, matrix_size, block_size):
                U_tile = wp.tile_load(
                    U,
                    shape=(block_size, block_size),
                    offset=(i, j),
                    storage="shared",
                )
                x_tile = wp.tile_view(
                    rhs_tile,
                    shape=(block_size, 1),
                    offset=(j, 0),
                )
                wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)
            U_tile = wp.tile_load(
                U,
                shape=(block_size, block_size),
                offset=(i, i),
                storage="shared",
            )
            wp.tile_upper_solve_inplace(U_tile, tmp_tile)

        wp.tile_store(x, rhs_tile, offset=(0, 0), bounds_check=True)

    return func


@cache
def make_indexed_cholesky_factor_solve_kernel(
    block_size: int,
    matrix_size: int,
    vector_size: int,
):
    factor_solve = create_indexed_cholesky_factor_solve_func(
        block_size, matrix_size, vector_size
    )

    @wp.kernel(
        module="unique",
        enable_backward=False,
        module_options={"enable_mathdx_gemm": False},
    )
    @expose_kernel_name
    def factor_solve_indexed_cholesky_kernel(
        M_in: wp.array2d[float],
        block_idx: wp.array[int],
        pad_identity: wp.array[float],
        y: wp.array3d[float],
        matrix_size_rt: int,
        nv_rt: int,
        U: wp.array3d[float],
        x: wp.array3d[float],
    ):
        worldid = wp.tid()
        wp.static(factor_solve)(
            M_in[worldid],
            block_idx,
            pad_identity,
            y[worldid],
            matrix_size_rt,
            nv_rt,
            U[worldid],
            x[worldid],
        )

    return factor_solve_indexed_cholesky_kernel


@cache
def make_indexed_cholesky_factor_multi_solve_kernel(
    block_size: int,
    matrix_size: int,
    vector_size: int,
    rhs_count: int,
):
    """Create the indexed blocked Cholesky solve for persistent multiple RHS."""
    block_area = block_size * block_size
    nblk = matrix_size // block_size

    @wp.kernel(
        module="unique",
        enable_backward=False,
        module_options={"enable_mathdx_gemm": False},
    )
    @expose_kernel_name
    def factor_multi_solve_indexed_cholesky_kernel(
        M_in: wp.array2d[float],
        block_idx: wp.array[int],
        pad_identity: wp.array[float],
        y: wp.array3d[float],
        active: wp.array[int],
        matrix_size_rt: int,
        nv_rt: int,
        U: wp.array3d[float],
        x: wp.array3d[float],
    ):
        worldid = wp.tid()
        if active[worldid] == 0:
            return
        rhs_tile = wp.tile_zeros(
            shape=(matrix_size, rhs_count),
            dtype=float,
            storage="shared",
        )
        active_rhs = wp.tile_load(
            y[worldid],
            shape=(vector_size, rhs_count),
            offset=(0, 0),
            storage="shared",
            bounds_check=False,
        )
        wp.tile_assign(rhs_tile, active_rhs, offset=(0, 0))

        for k in range(0, matrix_size_rt, block_size):
            kb = k // block_size
            end = k + block_size
            rhs_view = wp.tile_view(
                rhs_tile,
                shape=(block_size, rhs_count),
                offset=(k, 0),
            )
            idx = wp.tile_load(
                block_idx,
                shape=(block_area,),
                offset=((kb * nblk + kb) * block_area,),
                storage="shared",
            )
            flat = wp.tile_load_indexed(
                M_in[worldid],
                idx,
                shape=(block_area,),
                storage="shared",
            )
            diagonal = wp.tile_reshape(flat, (block_size, block_size))
            if end > nv_rt:
                pad_flat = wp.tile_load(
                    pad_identity,
                    shape=(block_area,),
                    offset=(0,),
                    storage="shared",
                )
                diagonal = wp.tile_map(
                    wp.add,
                    diagonal,
                    wp.tile_reshape(pad_flat, (block_size, block_size)),
                )

            for j in range(0, k, block_size):
                U_jk = wp.tile_load(
                    U[worldid],
                    shape=(block_size, block_size),
                    offset=(j, k),
                    storage="shared",
                )
                wp.tile_matmul(
                    wp.tile_transpose(U_jk),
                    U_jk,
                    diagonal,
                    alpha=-1.0,
                )
                y_j = wp.tile_view(
                    rhs_tile,
                    shape=(block_size, rhs_count),
                    offset=(j, 0),
                )
                wp.tile_matmul(
                    wp.tile_transpose(U_jk),
                    y_j,
                    rhs_view,
                    alpha=-1.0,
                )

            U_kk = wp.tile_cholesky(diagonal, fill_mode="upper")
            wp.tile_store(U[worldid], U_kk, offset=(k, k))
            wp.tile_lower_solve_inplace(wp.tile_transpose(U_kk), rhs_view)

            for i in range(end, matrix_size_rt, block_size):
                ib = i // block_size
                idx = wp.tile_load(
                    block_idx,
                    shape=(block_area,),
                    offset=((kb * nblk + ib) * block_area,),
                    storage="shared",
                )
                flat = wp.tile_load_indexed(
                    M_in[worldid],
                    idx,
                    shape=(block_area,),
                    storage="shared",
                )
                U_ki = wp.tile_reshape(flat, (block_size, block_size))
                for j in range(0, k, block_size):
                    U_jk = wp.tile_load(
                        U[worldid],
                        shape=(block_size, block_size),
                        offset=(j, k),
                        storage="shared",
                    )
                    U_ji = wp.tile_load(
                        U[worldid],
                        shape=(block_size, block_size),
                        offset=(j, i),
                        storage="shared",
                    )
                    wp.tile_matmul(
                        wp.tile_transpose(U_jk),
                        U_ji,
                        U_ki,
                        alpha=-1.0,
                    )
                wp.tile_lower_solve_inplace(wp.tile_transpose(U_kk), U_ki)
                wp.tile_store(U[worldid], U_ki, offset=(k, i))

        for i in range(matrix_size_rt - block_size, -1, -block_size):
            tmp = wp.tile_view(
                rhs_tile,
                shape=(block_size, rhs_count),
                offset=(i, 0),
            )
            for j in range(i + block_size, matrix_size_rt, block_size):
                U_ij = wp.tile_load(
                    U[worldid],
                    shape=(block_size, block_size),
                    offset=(i, j),
                    storage="shared",
                )
                x_j = wp.tile_view(
                    rhs_tile,
                    shape=(block_size, rhs_count),
                    offset=(j, 0),
                )
                wp.tile_matmul(U_ij, x_j, tmp, alpha=-1.0)
            U_ii = wp.tile_load(
                U[worldid],
                shape=(block_size, block_size),
                offset=(i, i),
                storage="shared",
            )
            wp.tile_upper_solve_inplace(U_ii, tmp)

        wp.tile_store(x[worldid], rhs_tile, offset=(0, 0), bounds_check=True)

    return factor_multi_solve_indexed_cholesky_kernel


@cache
def make_indexed_cholesky_multi_solve_kernel(
    block_size: int,
    matrix_size: int,
    vector_size: int,
    rhs_count: int,
):
    """Solve multiple RHS from an already computed upper Cholesky factor."""

    @wp.kernel(
        module="unique",
        enable_backward=False,
        module_options={"enable_mathdx_gemm": False},
    )
    @expose_kernel_name
    def multi_solve_indexed_cholesky_kernel(
        U: wp.array3d[float],
        y: wp.array3d[float],
        active: wp.array[int],
        x: wp.array3d[float],
    ):
        worldid = wp.tid()
        if active[worldid] == 0:
            return
        rhs_tile = wp.tile_zeros(
            shape=(matrix_size, rhs_count),
            dtype=float,
            storage="shared",
        )
        active_rhs = wp.tile_load(
            y[worldid],
            shape=(vector_size, rhs_count),
            offset=(0, 0),
            storage="shared",
            bounds_check=False,
        )
        wp.tile_assign(rhs_tile, active_rhs, offset=(0, 0))

        for k in range(0, matrix_size, block_size):
            rhs_view = wp.tile_view(
                rhs_tile,
                shape=(block_size, rhs_count),
                offset=(k, 0),
            )
            for j in range(0, k, block_size):
                U_jk = wp.tile_load(
                    U[worldid],
                    shape=(block_size, block_size),
                    offset=(j, k),
                    storage="shared",
                )
                y_j = wp.tile_view(
                    rhs_tile,
                    shape=(block_size, rhs_count),
                    offset=(j, 0),
                )
                wp.tile_matmul(
                    wp.tile_transpose(U_jk),
                    y_j,
                    rhs_view,
                    alpha=-1.0,
                )
            U_kk = wp.tile_load(
                U[worldid],
                shape=(block_size, block_size),
                offset=(k, k),
                storage="shared",
            )
            wp.tile_lower_solve_inplace(wp.tile_transpose(U_kk), rhs_view)

        for i in range(matrix_size - block_size, -1, -block_size):
            tmp = wp.tile_view(
                rhs_tile,
                shape=(block_size, rhs_count),
                offset=(i, 0),
            )
            for j in range(i + block_size, matrix_size, block_size):
                U_ij = wp.tile_load(
                    U[worldid],
                    shape=(block_size, block_size),
                    offset=(i, j),
                    storage="shared",
                )
                x_j = wp.tile_view(
                    rhs_tile,
                    shape=(block_size, rhs_count),
                    offset=(j, 0),
                )
                wp.tile_matmul(U_ij, x_j, tmp, alpha=-1.0)
            U_ii = wp.tile_load(
                U[worldid],
                shape=(block_size, block_size),
                offset=(i, i),
                storage="shared",
            )
            wp.tile_upper_solve_inplace(U_ii, tmp)

        wp.tile_store(x[worldid], rhs_tile, offset=(0, 0), bounds_check=True)

    return multi_solve_indexed_cholesky_kernel
