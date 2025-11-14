#include "../codelets.h"
#include "../../macros.h"
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
    auto in_p = (dahl_fp const*)in.ptr;

    auto out = STARPU_BLOCK_GET(buffers[1]);
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
    auto in_p = (dahl_fp const*)in.ptr;

    auto out = STARPU_VECTOR_GET(buffers[1]);
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
    auto ts = STARPU_TENSOR_GET(buffers[0]);
    auto ts_p = (dahl_fp*)ts.ptr;

    for (int t = 0; t < ts.nt; t++)
    {
        for (int z = 0; z < ts.nz; z++)
        {
            for (int y = 0; y < ts.ny; y++)
            {
                for (int x = 0; x < ts.nx; x++)
                {
                    ts_p[(t * ts.ldt) + (z * ts.ldz) + (y * ts.ldy) + x] = 0;
                }
            }
        }
    }
}

void tensor_accumulate(void *buffers[2], void *cl_arg)
{
	// dst tensor accumulator
    auto dst = STARPU_TENSOR_GET(buffers[0]);
    auto dst_p = (dahl_fp*)dst.ptr;

    // source tensor
    auto src = STARPU_TENSOR_GET(buffers[1]);
    auto src_p = (dahl_fp const*)src.ptr;

    assert(dst.nx == src.nx);
    assert(dst.ny == src.ny);
    assert(dst.nz == src.nz);
    assert(dst.nt == src.nt);
    assert(dst.ldy == src.ldy);
    assert(dst.ldz == src.ldz);
    assert(dst.ldt == src.ldt);

    for (int t = 0; t < dst.nt; t++)
    {
        for (int z = 0; z < dst.nz; z++)
        {
            for (int y = 0; y < dst.ny; y++)
            {
                for (int x = 0; x < dst.nx; x++)
                {
                    dst_p[(t * dst.ldt) + (z * dst.ldz) + (y * dst.ldy) + x] += 
                        src_p[(t * src.ldt) + (z * src.ldz) + (y * src.ldy) + x];
                }
            }
        }
    }
}
