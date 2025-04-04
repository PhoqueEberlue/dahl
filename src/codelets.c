#include "codelets.h"
#include "starpu_task_util.h"
#include "../include/dahl_types.h"
#include <assert.h>
#include <stdlib.h>


void matrix_cross_correlation(void* buffers[3], void* cl_arg)
{
    // Input matrix
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Kernel matrix
    size_t const kernel_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const kernel_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const kernel_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp const* const kernel = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output matrix
    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

void matrix_max_dahl_pooling(void* buffers[3], void* cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const mask_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const mask_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const mask_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* const mask = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

void matrix_backward_max_dahl_pooling(void *buffers[3], void *cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const mask_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const mask_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const mask_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp const* const mask = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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

void relu(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

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
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

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

void scal(void* buffers[2], void* cl_arg)
{
    dahl_fp factor;
    starpu_codelet_unpack_args(cl_arg, &factor);

    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

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
    dahl_fp const* const a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* const b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_nz = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const c_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* const c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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
    dahl_fp const* const a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* const b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_nz = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const c_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* const c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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
