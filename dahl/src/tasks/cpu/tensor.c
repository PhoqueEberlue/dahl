#include "../codelets.h"
#include "../macros.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../../include/dahl_types.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

void tensor_sum_t_axis(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_TENSOR_GET(buffers[0]);
    auto out = STARPU_BLOCK_GET(buffers[1]);
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx);
    assert(in.ny == out.ny);
    assert(in.nz == out.nz);

    for (int t = 0; t < in.nt; t++)
    {
        for (int z = 0; z < in.nz; z++)
        {
            for (int y = 0; y < in.ny; y++)
            {
                for (int x = 0; x < in.nx; x++)
                {
                    out_p[(z * out.ldz) + (y * out.ldy) + x] +=
                        in_p[(t * in.ldt) + (z * in.ldz) + (y * in.ldy) + x];
                }
            }
        }
    }
}

void tensor_sum_xyt_axes(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_TENSOR_GET(buffers[0]);
    auto out = STARPU_VECTOR_GET(buffers[1]);
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nz == out.nx);

    for (int t = 0; t < in.nt; t++)
    {
        for (int z = 0; z < in.nz; z++)
        {
            for (int y = 0; y < in.ny; y++)
            {
                for (int x = 0; x < in.nx; x++)
                {
                    out_p[z] += in_p[(t * in.ldt) + (z * in.ldz) + (y * in.ldy) + x];
                }
            }
        }
    }
}

void tensor_zero(void *buffers[1], void *cl_arg)
{
	// Tensor
    size_t const nx = STARPU_TENSOR_GET_NX(buffers[0]);
    size_t const ny = STARPU_TENSOR_GET_NY(buffers[0]);
    size_t const nz = STARPU_TENSOR_GET_NZ(buffers[0]);
    size_t const nt = STARPU_TENSOR_GET_NT(buffers[0]);
    size_t const ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    size_t const ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    size_t const ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
    dahl_fp* data = (dahl_fp*)STARPU_TENSOR_GET_PTR(buffers[0]);

    for (int t = 0; t < nt; t++)
    {
        for (int z = 0; z < nz; z++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[(t * ldt) + (z * ldz) + (y * ldy) + x] = 0;
                }
            }
        }
    }
}

void tensor_accumulate(void *buffers[2], void *cl_arg)
{
	// dst tensor accumulator
    size_t const dst_nx = STARPU_TENSOR_GET_NX(buffers[0]);
    size_t const dst_ny = STARPU_TENSOR_GET_NY(buffers[0]);
    size_t const dst_nz = STARPU_TENSOR_GET_NZ(buffers[0]);
    size_t const dst_nt = STARPU_TENSOR_GET_NT(buffers[0]);
    size_t const dst_ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    size_t const dst_ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    size_t const dst_ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
    dahl_fp* dst = (dahl_fp*)STARPU_TENSOR_GET_PTR(buffers[0]);

    // source tensor
    size_t const src_nx = STARPU_TENSOR_GET_NX(buffers[1]);
    size_t const src_ny = STARPU_TENSOR_GET_NY(buffers[1]);
    size_t const src_nz = STARPU_TENSOR_GET_NZ(buffers[1]);
    size_t const src_nt = STARPU_TENSOR_GET_NT(buffers[1]);
    size_t const src_ldy = STARPU_TENSOR_GET_LDY(buffers[1]);
    size_t const src_ldz = STARPU_TENSOR_GET_LDZ(buffers[1]);
    size_t const src_ldt = STARPU_TENSOR_GET_LDT(buffers[1]);
    dahl_fp const* src = (dahl_fp*)STARPU_TENSOR_GET_PTR(buffers[1]);

    assert(dst_nx == src_nx);
    assert(dst_ny == src_ny);
    assert(dst_nz == src_nz);
    assert(dst_nt == src_nt);
    assert(dst_ldy == src_ldy);
    assert(dst_ldz == src_ldz);
    assert(dst_ldt == src_ldt);

    for (int t = 0; t < dst_nt; t++)
    {
        for (int z = 0; z < dst_nz; z++)
        {
            for (int y = 0; y < dst_ny; y++)
            {
                for (int x = 0; x < dst_nx; x++)
                {
                    dst[(t * dst_ldt) + (z * dst_ldz) + (y * dst_ldy) + x] += 
                        src[(t * src_ldt) + (z * src_ldz) + (y * src_ldy) + x];
                }
            }
        }
    }
}
