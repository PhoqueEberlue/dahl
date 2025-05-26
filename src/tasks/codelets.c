#include "codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../include/dahl_types.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


void matrix_cross_correlation(void* buffers[3], void* cl_arg)
{
    // Input matrix
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Kernel matrix
    size_t const kernel_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const kernel_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const kernel_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp const* kernel = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output matrix
    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const mask_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const mask_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const mask_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* mask = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(out_nx == in_nx / pool_size);
    assert(out_ny == in_ny / pool_size);

    assert(in_nx == mask_nx);
    assert(in_ny == mask_ny);

    // Loop through i,j on axes x,y of matrix `out`
    for (size_t j = 0; j < out_ny; j++)
    {
        for (size_t i = 0; i < out_nx; i++)
        {
            size_t start_l = j * pool_size;
            size_t start_k = i * pool_size;
            size_t end_l = start_l + pool_size;
            size_t end_k = start_k + pool_size;

            dahl_fp current_max = 0;
            size_t current_max_y = 0;
            size_t current_max_x = 0;

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

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const mask_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const mask_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const mask_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp const* mask = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

            // Loop through k,l on axes x,y of matrix `out`
            for (size_t l = start_l; l < end_l; l++)
            {
                for (size_t k = start_k; k < end_k; k++)
                {
                    dahl_fp in_value = in[(j * in_ld) + i];
                    dahl_fp mask_value = mask[(l * mask_ld) + k];

                    out[(l * out_ld) + k] = in_value * mask_value;
                }
            }
        }
    }
}

void matrix_matrix_product(void* buffers[3], void* cl_arg)
{
    size_t const a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const a_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

void relu(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);
    assert(in_nz == out_nz);

    for (int i = 0; i < in_nx*in_ny*in_nz; i++)
    {
        if (in[i] < 0.0F)
        {
            out[i] = 0.0F;
        }
    }
}

void block_sum_z_axis(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);

    for (int z = 0; z < in_nz; z++)
    {
        for (int y = 0; y < in_ny; y++)
        {
            for (int x = 0; x < in_nx; x++)
            {
                // FIX ME
                out[(y * out_ld) + x] += in[(z * in_ldz) + (y * in_ldy) + x];
            }
        }
    }
}

void matrix_sum_y_axis(void* buffers[2], void* cl_arg)
{
    // input matrix
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // output vector
    size_t const out_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_len);

    for (int x = 0; x < in_nx; x++)
    {
        for (int y = 0; y < in_ny; y++)
        {
            out[x] += in[(y * in_ld) + x];
        }
    }
}

void scal(void* buffers[2], void* cl_arg)
{
    dahl_fp factor;
    starpu_codelet_unpack_args(cl_arg, &factor);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);
    assert(in_nz == out_nz);

    for (int i = 0; i < in_nx*in_ny*in_nz; i++)
    {
        out[i] = in[i] * factor;
    }
}

void sub(void* buffers[3], void* cl_arg)
{
    size_t const a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_nz = STARPU_BLOCK_GET_NZ(buffers[2]);
    size_t const c_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const c_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(a_nx == b_nx && a_nx == c_nx);
    assert(a_ny == b_ny && a_ny == c_ny);
    assert(a_nz == b_nz && a_nz == c_nz);
    assert(a_ldy == b_ldy && a_ldy == c_ldy);
    assert(a_ldz == b_ldz && a_ldz == c_ldz);

    for (int z = 0; z < a_nz; z++)
    {
        for (int y = 0; y < a_ny; y++)
        {
            for (int x = 0; x < a_nx; x++)
            {
                dahl_fp value_a = a[(z * a_ldz) + (y * a_ldy) + x];
                dahl_fp value_b = b[(z * b_ldz) + (y * b_ldy) + x];
                c[(z * c_ldz) + (y * c_ldy) + x] = value_a - value_b;
            }
        }
    }
}

void add(void* buffers[3], void* cl_arg)
{
    size_t const a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_nz = STARPU_BLOCK_GET_NZ(buffers[2]);
    size_t const c_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const c_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(a_nx == b_nx && a_nx == c_nx);
    assert(a_ny == b_ny && a_ny == c_ny);
    assert(a_nz == b_nz && a_nz == c_nz);
    assert(a_ldy == b_ldy && a_ldy == c_ldy);
    assert(a_ldz == b_ldz && a_ldz == c_ldz);

    for (int z = 0; z < a_nz; z++)
    {
        for (int y = 0; y < a_ny; y++)
        {
            for (int x = 0; x < a_nx; x++)
            {
                dahl_fp value_a = a[(z * a_ldz) + (y * a_ldy) + x];
                dahl_fp value_b = b[(z * b_ldz) + (y * b_ldy) + x];
                c[(z * c_ldz) + (y * c_ldy) + x] = value_a + value_b;
            }
        }
    }
}

void vector_softmax(void* buffers[2], void* cl_arg)
{
    size_t const in_len = STARPU_BLOCK_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_len == out_len);

    dahl_fp max_value = 0.0F;

    // Getting max value
    for (size_t i = 0; i < in_len; i++)
    {
        if (in[i] > max_value)
        {
            max_value = in[i];
        }
    }

    dahl_fp sum_values = 0.0F;

    // Shifting by the max value, computing exponent for each element, and summing
    for (size_t i = 0; i < in_len; i++)
    {
        out[i] = exp(in[i] - max_value);
        sum_values += out[i];
    }

    // Computing the probabilities
    for (size_t i = 0; i < in_len; i++)
    {
        out[i] = out[i] / sum_values;
    }
}

void vector_dot_product(void* buffers[2], void* cl_arg)
{
    dahl_fp *res_p;
    starpu_codelet_unpack_args(cl_arg, &res_p);

    size_t const a_len = STARPU_BLOCK_GET_NX(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(a_len == b_len);

    dahl_fp res = 0;

    for (size_t i = 0; i < a_len; i++)
    {
        res += a[i] * b[i];
    }

    // Pass return value as a pointer within the arguments of the codelet
    *res_p = res;
}

void vector_diag(void* buffers[2], void* cl_arg)
{
    // Input vector
    size_t const in_len = STARPU_BLOCK_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Output matrix
    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_len == out_nx);
    assert(in_len == out_ny);

    for (size_t i = 0; i < in_len; i++)
    {
        // Copy the vector's elements in a diagonal manner into the matrix
        out[(i * out_ld) + i] = in[i];
    }
}

void add_value(void* buffers[2], void* cl_arg)
{
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &value);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);
    assert(in_nz == out_nz);

    for (int i = 0; i < in_nx*in_ny*in_nz; i++)
    {
        out[i] = in[i] + value;
    }
}

void sub_value(void* buffers[2], void* cl_arg)
{
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &value);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);
    assert(in_nz == out_nz);

    for (int i = 0; i < in_nx*in_ny*in_nz; i++)
    {
        out[i] = in[i] - value;
    }
}

void matrix_vector_product(void* buffers[3], void* cl_arg)
{
    // Input matrix
    size_t const mat_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const mat_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const mat_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* mat = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Input vector
    size_t const vec_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp const* vec = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output vector
    size_t const out_len = STARPU_BLOCK_GET_NX(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

void clip(void* buffers[2], void* cl_arg)
{
    dahl_fp min;
    dahl_fp max;
    starpu_codelet_unpack_args(cl_arg, &min, &max);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);
    assert(in_nz == out_nz);

    for (int i = 0; i < in_nx*in_ny*in_nz; i++)
    {
        if (in[i] > max)
        {
            out[i] = max;
        }
        else if (in[i] < min)
        {
            out[i] = min;
        }
        else
        {
            out[i] = in[i];
        }
    }
}

void vector_cross_entropy_loss(void* buffers[2], void* cl_arg)
{
    dahl_fp *res_p;
    starpu_codelet_unpack_args(cl_arg, &res_p);

    // Predictions vector
    size_t const pred_len = STARPU_BLOCK_GET_NX(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Targets vector
    size_t const targ_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(pred_len == targ_len);

    dahl_fp loss = 0;

    for (size_t i = 0; i < pred_len; i++)
    {
        loss += (targ[i] * log(pred[i]));
    }

    // Divide by the number of classes and reverse the sign
    loss = - (loss / (dahl_fp)pred_len);

    // Pass return value as a pointer within the arguments of the codelet
    *res_p = loss;
}

void vector_cross_entropy_loss_gradient(void* buffers[3], void* cl_arg)
{
    // Predictions vector
    size_t const pred_len = STARPU_BLOCK_GET_NX(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Targets vector
    size_t const targ_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output vector
    size_t const out_len = STARPU_BLOCK_GET_NX(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(pred_len == targ_len);
    assert(pred_len == out_len);

    for (size_t i = 0; i < out_len; i++)
    {
        out[i] = (-targ[i]) / (pred[i] + 1e-7F) / (dahl_fp)pred_len;
    }
}

void matrix_transpose(void* buffers[2], void* cl_arg)
{
    // Input matrix
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Output matrix
    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

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
