#include "../codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../../include/dahl_types.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

void block_sum_z_axis(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);

    for (int z = 0; z < in_nz; z++)
    {
        for (int y = 0; y < in_ny; y++)
        {
            for (int x = 0; x < in_nx; x++)
            {
                out[(y * out_ld) + x] += in[(z * in_ldz) + (y * in_ldy) + x];
            }
        }
    }
}

void block_sum_y_axis(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_nz == out_ny);

    for (int y = 0; y < in_ny; y++)
    {
        for (int z = 0; z < in_nz; z++)
        {
            for (int x = 0; x < in_nx; x++)
            {
                out[(z * out_ld) + x] += in[(z * in_ldz) + (y * in_ldy) + x];
            }
        }
    }
}

void block_sum_xy_axes(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    assert(in_nz == out_len);

    for (int z = 0; z < in_nz; z++)
    {
        for (int y = 0; y < in_ny; y++)
        {
            for (int x = 0; x < in_nx; x++)
            {
                out[z] += in[(z * in_ldz) + (y * in_ldy) + x];
            }
        }
    }
}

void block_add_padding(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(out_nx >= in_nx && out_ny >= in_ny && out_nz >= in_nz);

    size_t diff_z = (out_nz - in_nz) / 2;
    size_t diff_y = (out_ny - in_ny) / 2;
    size_t diff_x = (out_nx - in_nx) / 2;

    for (size_t z = 0; z < in_nz; z++)
    {
        for (size_t y = 0; y < in_ny; y++)
        {
            for (size_t x = 0; x < in_nx; x++)
            {
                dahl_fp value = in[(z * in_ldz) + (y * in_ldy) + x];
                out[((z + diff_z) * out_ldz) + ((y + diff_y) * out_ldy) + (x + diff_x)] = value;
            }
        }

    }
}

void block_zero(void *buffers[1], void *cl_arg)
{
	// Block
    size_t const nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp* data = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    for (int z = 0; z < nz; z++)
    {
        for (int y = 0; y < ny; y++)
        {
            for (int x = 0; x < nx; x++)
            {
                data[(z * ldz) + (y * ldy) + x] = 0;
            }
        }
    }
}

void block_accumulate(void *buffers[2], void *cl_arg)
{
	// dst block accumulator
    size_t const dst_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const dst_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const dst_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const dst_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const dst_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp* dst = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // source block
    size_t const src_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const src_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const src_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const src_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const src_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* src = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(dst_nx == src_nx);
    assert(dst_ny == src_ny);
    assert(dst_nz == src_nz);
    assert(dst_ldy == src_ldy);
    assert(dst_ldz == src_ldz);

    for (int z = 0; z < dst_nz; z++)
    {
        for (int y = 0; y < dst_ny; y++)
        {
            for (int x = 0; x < dst_nx; x++)
            {
                dst[(z * dst_ldz) + (y * dst_ldy) + x] += 
                    src[(z * src_ldz) + (y * src_ldy) + x];
            }
        }
    }
}
