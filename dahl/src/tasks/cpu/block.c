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
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    auto out = STARPU_MATRIX_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx);
    assert(in.ny == out.ny);

    for (size_t z = 0; z < in.nz; z++)
    {
        for (size_t y = 0; y < in.ny; y++)
        {
            for (size_t x = 0; x < in.nx; x++)
            {
                out_p[(y * out.ld) + x] += in_p[(z * in.ldz) + (y * in.ldy) + x];
            }
        }
    }
}

void block_sum_y_axis(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    auto out = STARPU_MATRIX_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx);
    assert(in.nz == out.ny);

    for (size_t y = 0; y < in.ny; y++)
    {
        for (size_t z = 0; z < in.nz; z++)
        {
            for (size_t x = 0; x < in.nx; x++)
            {
                out_p[(z * out.ld) + x] += in_p[(z * in.ldz) + (y * in.ldy) + x];
            }
        }
    }
}

void block_sum_xy_axes(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    auto out = STARPU_VECTOR_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nz == out.nx);

    for (size_t z = 0; z < in.nz; z++)
    {
        for (size_t y = 0; y < in.ny; y++)
        {
            for (size_t x = 0; x < in.nx; x++)
            {
                out_p[z] += in_p[(z * in.ldz) + (y * in.ldy) + x];
            }
        }
    }
}

// void block_add_padding(void* buffers[2], void* cl_arg)
// {
//     size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
//     size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
//     size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
//     size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
//     size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
//     dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);
// 
//     size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
//     size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
//     size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
//     size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
//     size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
//     dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);
// 
//     assert(out_nx >= in_nx && out_ny >= in_ny && out_nz >= in_nz);
// 
//     size_t diff_z = (out_nz - in_nz) / 2;
//     size_t diff_y = (out_ny - in_ny) / 2;
//     size_t diff_x = (out_nx - in_nx) / 2;
// 
//     for (size_t z = 0; z < in_nz; z++)
//     {
//         for (size_t y = 0; y < in_ny; y++)
//         {
//             for (size_t x = 0; x < in_nx; x++)
//             {
//                 dahl_fp value = in[(z * in_ldz) + (y * in_ldy) + x];
//                 out[((z + diff_z) * out_ldz) + ((y + diff_y) * out_ldy) + (x + diff_x)] = value;
//             }
//         }
// 
//     }
// }

void block_zero(void *buffers[1], void *cl_arg)
{
    auto bl = STARPU_BLOCK_GET(buffers[0]);
    auto bl_p = (dahl_fp*)bl.ptr;

    for (size_t z = 0; z < bl.nz; z++)
    {
        for (size_t y = 0; y < bl.ny; y++)
        {
            for (size_t x = 0; x < bl.nx; x++)
            {
                bl_p[(z * bl.ldz) + (y * bl.ldy) + x] = 0;
            }
        }
    }
}

void block_accumulate(void *buffers[2], void *cl_arg)
{
	// dst block accumulator
    auto dst = STARPU_BLOCK_GET(buffers[0]);
    auto dst_p = (dahl_fp*)dst.ptr;

    // source block
    auto src = STARPU_BLOCK_GET(buffers[1]);
    auto src_p = (dahl_fp const*)src.ptr;

    assert(dst.nx == src.nx);
    assert(dst.ny == src.ny);
    assert(dst.nz == src.nz);
    assert(dst.ldy == src.ldy);
    assert(dst.ldz == src.ldz);

    for (size_t z = 0; z < dst.nz; z++)
    {
        for (size_t y = 0; y < dst.ny; y++)
        {
            for (size_t x = 0; x < dst.nx; x++)
            {
                dst_p[(z * dst.ldz) + (y * dst.ldy) + x] += 
                    src_p[(z * src.ldz) + (y * src.ldy) + x];
            }
        }
    }
}
