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
    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Kernel matrix
    size_t const kernel_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const kernel_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const kernel_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* kernel = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(out_nx == in_nx - kernel_nx + 1);
    assert(out_ny == in_ny - kernel_ny + 1);

    // loop through i,j on axes x,y of the output matrix
    for (size_t j = 0; j < out_ny; j++)
    {
        for (size_t i = 0; i < out_nx; i++)
        {
            dahl_fp cell_res = 0.0F;

            // loop through k,l on axes x,y of the kernel
            for (size_t l = 0; l < kernel_ny; l++)
            {
                for (size_t k = 0; k < kernel_nx; k++)
                {
                    dahl_fp in_value = in[((l + j) * in_ld) + k + i];
                    dahl_fp kernel_value = kernel[(l * kernel_ld) + k];
                    
                    cell_res += in_value * kernel_value;
                }
            }

            out[(j * out_ld) + i] = cell_res;
        }
    }
}

void matrix_max_pooling(void* buffers[3], void* cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    size_t const mask_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const mask_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const mask_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* mask = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(out_nx == in_nx / pool_size);
    assert(out_ny == in_ny / pool_size);

    assert(in_nx == mask_nx);
    assert(in_ny == mask_ny);

    // Loop through i,j on axes x,y of matrix `out`
    for (size_t j = 0; j < out_ny; j++)
    {
        for (size_t i = 0; i < out_nx; i++)
        {
            // Compute pooling window bounds
            size_t start_l = j * pool_size;
            size_t start_k = i * pool_size;
            size_t end_l = start_l + pool_size;
            size_t end_k = start_k + pool_size;

            // Set max value and index to first element to handle cases with negative numbers
            dahl_fp current_max = in[(start_l * in_ld) + start_k];
            size_t current_max_y = start_l;
            size_t current_max_x = start_k;

            // Loop through k,l on axes x,y of matrix `in`
            for (size_t l = start_l; l < end_l; l++)
            {
                for (size_t k = start_k; k < end_k; k++)
                {
                    dahl_fp in_value = in[(l * in_ld) + k];

                    if (in_value > current_max)
                    {
                        current_max = in_value;
                        current_max_y = l;
                        current_max_x = k;
                    }
                }
            }

            out[(j * out_ld) + i] = current_max;
            mask[(current_max_y * mask_ld) + current_max_x] = 1.0F;
        }
    }
}

void matrix_backward_max_pooling(void *buffers[3], void *cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    size_t const mask_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const mask_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const mask_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* mask = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(in_nx == out_nx / pool_size);
    assert(in_ny == out_ny / pool_size);

    assert(out_nx == mask_nx);
    assert(out_ny == mask_ny);

    // Loop through i,j on axes x,y of matrix `in`
    for (size_t j = 0; j < in_ny; j++)
    {
        for (size_t i = 0; i < in_nx; i++)
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
                    dahl_fp in_value = in[(j * in_ld) + i];
                    dahl_fp mask_value = mask[(l * mask_ld) + k];

                    // If the current value of the mask mask is 0, the result is ignored.
                    // I'm not sure if its better to check that conditionnaly, or to always write a result.
                    out[(l * out_ld) + k] = in_value * mask_value;
                }
            }
        }
    }
}

void matrix_matrix_product(void* buffers[3], void* cl_arg)
{
    size_t const a_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const a_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const b_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const c_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* c = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(a_nx == b_ny);
    assert(c_ny == a_ny);
    assert(c_nx == b_nx);

    // Loop through (x,y) of c
    for (size_t y = 0; y < c_ny; y++)
    {
        for (size_t x = 0; x < c_nx; x++)
        {
            for (size_t i = 0; i < a_nx; i++)
            {
                dahl_fp val_a = a[(y * a_ld) + i];
                dahl_fp val_b = b[(i * b_ld) + x];

                c[(y * c_ld) + x] += val_a * val_b;
            }
        }
    }
}

void matrix_sum_y_axis(void* buffers[2], void* cl_arg)
{
    // input matrix
    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // output vector
    size_t const out_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    assert(in_nx == out_len);

    for (int x = 0; x < in_nx; x++)
    {
        for (int y = 0; y < in_ny; y++)
        {
            out[x] += in[(y * in_ld) + x];
        }
    }
}

void matrix_vector_product(void* buffers[3], void* cl_arg)
{
    // Input matrix
    size_t const mat_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const mat_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const mat_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* mat = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Input vector
    size_t const vec_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp const* vec = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    // Output vector
    size_t const out_len = STARPU_VECTOR_GET_NX(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[2]);

    assert(vec_len == mat_nx);
    assert(out_len == mat_ny);

    // Loop through x,y of the matrix
    for (size_t y = 0; y < mat_ny; y++)
    {
        for (size_t x = 0; x < mat_nx; x++)
        {
            out[y] += vec[x] * mat[(y * mat_ld) + x];
        }
    }
}

void matrix_transpose(void* buffers[2], void* cl_arg)
{
    // Input matrix
    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(in_nx == out_ny);
    assert(in_ny == out_nx);

    for (size_t y = 0; y < in_ny; y++)
    {
        for (size_t x = 0; x < in_nx; x++)
        {
            out[(x * out_ld) + y] = in[(y * in_ld) + x];
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
    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);

    for (size_t y = 0; y < in_ny; y++)
    {
        for (size_t x = 0; x < in_nx; x++)
        {
            size_t y_rot = (out_ny - 1 - y);
            size_t x_rot = (out_nx - 1 - x);
            out[(y_rot * out_ld) + x_rot] = in[(y * in_ld) + x];
        }
    }
}

void matrix_zero(void *buffers[1], void *cl_arg)
{
	// Matrix
    size_t const nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp* data = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            data[(y * ld) + x] = 0;
        }
    }
}

void matrix_accumulate(void *buffers[2], void *cl_arg)
{
	// dst matrix accumulator
    size_t const dst_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const dst_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const dst_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp* dst = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // source matrix
    size_t const src_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const src_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const src_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* src = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(dst_nx == src_nx);
    assert(dst_ny == src_ny);
    assert(dst_ld == src_ld);

    for (size_t y = 0; y < dst_ny; y++)
    {
        for (size_t x = 0; x < dst_nx; x++)
        {
            dst[(y * dst_ld) + x] += src[(y * src_ld) + x];
        }
    }
}
