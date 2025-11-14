#include "../codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../../include/dahl_types.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <sys/mman.h>

void matrix_cross_correlation(void* buffers[3], void* cl_arg)
{
    // Input matrix
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    // Kernel matrix
    auto ker = STARPU_MATRIX_GET(buffers[1]);
    auto ker_p = (dahl_fp const*)ker.ptr;

    // Output matrix
    auto out = STARPU_MATRIX_GET(buffers[2]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(out.nx == in.nx - ker.nx + 1);
    assert(out.ny == in.ny - ker.ny + 1);

    // loop through i,j on axes x,y of the output matrix
    for (size_t j = 0; j < out.ny; j++)
    {
        for (size_t i = 0; i < out.nx; i++)
        {
            dahl_fp cell_res = 0.0F;

            // loop through k,l on axes x,y of the kernel
            for (size_t l = 0; l < ker.ny; l++)
            {
                for (size_t k = 0; k < ker.nx; k++)
                {
                    dahl_fp in_value = in_p[((l + j) * in.ld) + k + i];
                    dahl_fp kernel_value = ker_p[(l * ker.ld) + k];
                    
                    cell_res += in_value * kernel_value;
                }
            }

            out_p[(j * out.ld) + i] = cell_res;
        }
    }
}

void matrix_max_pooling(void* buffers[3], void* cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    auto mask = STARPU_MATRIX_GET(buffers[1]);
    auto mask_p = (dahl_fp*)mask.ptr;

    auto out = STARPU_MATRIX_GET(buffers[2]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(out.nx == in.nx / pool_size);
    assert(out.ny == in.ny / pool_size);

    assert(in.nx == mask.nx);
    assert(in.ny == mask.ny);

    // Loop through i,j on axes x,y of matrix `out`
    for (size_t j = 0; j < out.ny; j++)
    {
        for (size_t i = 0; i < out.nx; i++)
        {
            // Compute pooling window bounds
            size_t start_l = j * pool_size;
            size_t start_k = i * pool_size;
            size_t end_l = start_l + pool_size;
            size_t end_k = start_k + pool_size;

            // Set max value and index to first element to handle cases with negative numbers
            dahl_fp current_max = in_p[(start_l * in.ld) + start_k];
            size_t current_max_y = start_l;
            size_t current_max_x = start_k;

            // Loop through k,l on axes x,y of matrix `in`
            for (size_t l = start_l; l < end_l; l++)
            {
                for (size_t k = start_k; k < end_k; k++)
                {
                    dahl_fp in_value = in_p[(l * in.ld) + k];

                    if (in_value > current_max)
                    {
                        current_max = in_value;
                        current_max_y = l;
                        current_max_x = k;
                    }
                }
            }

            out_p[(j * out.ld) + i] = current_max;
            mask_p[(current_max_y * mask.ld) + current_max_x] = 1.0F;
        }
    }
}

void matrix_backward_max_pooling(void *buffers[3], void *cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    auto mask = STARPU_MATRIX_GET(buffers[1]);
    auto mask_p = (dahl_fp const*)mask.ptr;

    auto out = STARPU_MATRIX_GET(buffers[2]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx / pool_size);
    assert(in.ny == out.ny / pool_size);

    assert(out.nx == mask.nx);
    assert(out.ny == mask.ny);

    // Loop through i,j on axes x,y of matrix `in`
    for (size_t j = 0; j < in.ny; j++)
    {
        for (size_t i = 0; i < in.nx; i++)
        {
            size_t start_l = j * pool_size;
            size_t start_k = i * pool_size;
            size_t end_l = start_l + pool_size;
            size_t end_k = start_k + pool_size;

            // Loop through k,l on axes x,y of matrix `out` and `mask`
            for (size_t l = start_l; l < end_l; l++)
            {
                for (size_t k = start_k; k < end_k; k++)
                {
                    dahl_fp in_value = in_p[(j * in.ld) + i];
                    dahl_fp mask_value = mask_p[(l * mask.ld) + k];

                    // If the current value of the mask mask is 0, the result is ignored.
                    // I'm not sure if its better to check that conditionnaly, or to always write a result.
                    out_p[(l * out.ld) + k] = in_value * mask_value;
                }
            }
        }
    }
}

void matrix_matrix_product(void* buffers[3], void* cl_arg)
{
    auto a = STARPU_MATRIX_GET(buffers[0]);
    auto a_p = (dahl_fp const*)a.ptr;

    auto b = STARPU_MATRIX_GET(buffers[1]);
    auto b_p = (dahl_fp const*)b.ptr;

    auto c = STARPU_MATRIX_GET(buffers[2]);
    auto c_p = (dahl_fp*)c.ptr;

    assert(a.nx == b.ny);
    assert(c.ny == a.ny);
    assert(c.nx == b.nx);

    // Loop through (x,y) of c
    for (size_t y = 0; y < c.ny; y++)
    {
        for (size_t x = 0; x < c.nx; x++)
        {
            for (size_t i = 0; i < a.nx; i++)
            {
                dahl_fp val_a = a_p[(y * a.ld) + i];
                dahl_fp val_b = b_p[(i * b.ld) + x];

                c_p[(y * c.ld) + x] += val_a * val_b;
            }
        }
    }
}

void matrix_sum_y_axis(void* buffers[2], void* cl_arg)
{
    // input matrix
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    // output vector
    auto out = STARPU_VECTOR_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx);

    for (int x = 0; x < in.nx; x++)
    {
        for (int y = 0; y < in.ny; y++)
        {
            out_p[x] += in_p[(y * in.ld) + x];
        }
    }
}

void matrix_vector_product(void* buffers[3], void* cl_arg)
{
    // Input matrix
    auto mat = STARPU_MATRIX_GET(buffers[0]);
    auto mat_p = (dahl_fp const*)mat.ptr;

    // Input vector
    auto vec = STARPU_VECTOR_GET(buffers[1]);
    auto vec_p = (dahl_fp const*)vec.ptr;

    // Output vector
    auto out = STARPU_VECTOR_GET(buffers[2]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(vec.nx == mat.nx);
    assert(out.nx == mat.ny);

    // Loop through x,y of the matrix
    for (size_t y = 0; y < mat.ny; y++)
    {
        for (size_t x = 0; x < mat.nx; x++)
        {
            out_p[y] += vec_p[x] * mat_p[(y * mat.ld) + x];
        }
    }
}

void matrix_transpose(void* buffers[2], void* cl_arg)
{
    // Input matrix
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    // Output matrix
    auto out = STARPU_MATRIX_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.ny);
    assert(in.ny == out.nx);

    for (size_t y = 0; y < in.ny; y++)
    {
        for (size_t x = 0; x < in.nx; x++)
        {
            out_p[(x * out.ld) + y] = in_p[(y * in.ld) + x];
        }
    }
}

void matrix_resize(void* buffers[1], void* cl_arg)
{
    size_t new_nx;
    size_t new_ny;
    size_t new_ld;
    starpu_codelet_unpack_args(cl_arg, &new_nx, &new_ny, &new_ld);

    STARPU_MATRIX_SET_NX(buffers[0], new_nx);
    STARPU_MATRIX_SET_NY(buffers[0], new_ny);
    STARPU_MATRIX_SET_LD(buffers[0], new_ld);
}

void matrix_rotate_180(void* buffers[2], void* cl_arg)
{
    // Input matrix
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    // Output matrix
    auto out = STARPU_MATRIX_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx);
    assert(in.ny == out.ny);

    for (size_t y = 0; y < in.ny; y++)
    {
        for (size_t x = 0; x < in.nx; x++)
        {
            size_t y_rot = (out.ny - 1 - y);
            size_t x_rot = (out.nx - 1 - x);
            out_p[(y_rot * out.ld) + x_rot] = in_p[(y * in.ld) + x];
        }
    }
}

void matrix_zero(void *buffers[1], void *cl_arg)
{
	// Matrix
    auto mat = STARPU_MATRIX_GET(buffers[0]);
    auto mat_p = (dahl_fp*)mat.ptr;

    for (int y = 0; y < mat.ny; y++)
    {
        for (int x = 0; x < mat.nx; x++)
        {
            mat_p[(y * mat.ld) + x] = 0;
        }
    }
}

void matrix_accumulate(void *buffers[2], void *cl_arg)
{
	// dst matrix accumulator
    auto dst = STARPU_MATRIX_GET(buffers[0]);
    auto dst_p = (dahl_fp*)dst.ptr;

    // source matrix
    auto src = STARPU_MATRIX_GET(buffers[1]);
    auto src_p = (dahl_fp const*)src.ptr;

    assert(dst.nx == src.nx);
    assert(dst.ny == src.ny);
    assert(dst.ld == src.ld);

    for (size_t y = 0; y < dst.ny; y++)
    {
        for (size_t x = 0; x < dst.nx; x++)
        {
            dst_p[(y * dst.ld) + x] += src_p[(y * src.ld) + x];
        }
    }
}
