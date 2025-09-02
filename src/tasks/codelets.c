#include "codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../include/dahl_types.h"
#include "unistd.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

// ---------------------------------------- TENSOR ----------------------------------------
void tensor_sum_t_axis(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_TENSOR_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_TENSOR_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_TENSOR_GET_NZ(buffers[0]);
    size_t const in_nt = STARPU_TENSOR_GET_NT(buffers[0]);
    size_t const in_ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    size_t const in_ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_TENSOR_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_nx == out_nx);
    assert(in_ny == out_ny);
    assert(in_nz == out_nz);

    for (int t = 0; t < in_nt; t++)
    {
        for (int z = 0; z < in_nz; z++)
        {
            for (int y = 0; y < in_ny; y++)
            {
                for (int x = 0; x < in_nx; x++)
                {
                    out[(z * out_ldz) + (y * out_ldy) + x] +=
                        in[(t * in_ldt) + (z * in_ldz) + (y * in_ldy) + x];
                }
            }
        }
    }
}

// ---------------------------------------- BLOCK ----------------------------------------
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
                // FIX ME
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

// ---------------------------------------- MATRIX ----------------------------------------
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

// ---------------------------------------- VECTOR ----------------------------------------
void vector_softmax(void* buffers[2], void* cl_arg)
{
    size_t const in_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    size_t const out_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

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

void vector_dot_product(void* buffers[3], void* cl_arg)
{
    size_t const a_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    size_t const b_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    dahl_fp* c = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(a_len == b_len);

    for (size_t i = 0; i < a_len; i++)
    {
        *c += a[i] * b[i];
    }
}

void vector_diag(void* buffers[2], void* cl_arg)
{
    // Input vector
    size_t const in_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(in_len == out_nx);
    assert(in_len == out_ny);

    for (size_t i = 0; i < in_len; i++)
    {
        // Copy the vector's elements in a diagonal manner into the matrix
        out[(i * out_ld) + i] = in[i];
    }
}

void vector_to_matrix(void* buffers[2], void* cl_arg)
{
    size_t const in_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(out_nx * out_ny == in_len);
    
    for (size_t i = 0; i < in_len; i++)
        out[i] = in[i];
}

// ---------------------------------------- ANY ----------------------------------------
// Get the ptr of any StarPU data type. Does not perform any check.
// This works because ptr is always the second field in the struct for vector, matrix, block and tensor,
// so it does not matter what we cast `interface` into. 
// This may be risky though, especially if the field order changes...
#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)

void relu(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (in[i] < 0.0F)
        {
            out[i] = 0.0F;
        }
        else 
        {
            out[i] = in[i];
        }
    }
}

void scal(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp factor;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &factor);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = in[i] * factor;
    }
}

void power(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp power;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &power);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = pow(in[i], power);
    }
}

void sub(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* a = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp const* b = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    dahl_fp* c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        c[i] = a[i] - b[i];
    }
}

void add(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* a = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp const* b = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    dahl_fp* c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void add_value(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &value);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = in[i] + value;
    }
}

void clip(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp min;
    dahl_fp max;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &min, &max);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
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

void sum(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        *out += in[i];
    }
}

void mean(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

    dahl_fp sum = 0.0F;

    for (size_t i = 0; i < nb_elem; i++)
    {
        sum += in[i];
    }

    *out = sum / (dahl_fp)nb_elem;
}

void fill(void* buffers[1], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &value);

    dahl_fp* buf = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        buf[i] = value;
    }
}

// For debug purposes
void wait(void* buffers[1], void* cl_arg)
{
    unsigned int duration;
    starpu_codelet_unpack_args(cl_arg, &duration);
    usleep(duration);
}

void copy(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = in[i];
    }
}

// ---------------------------------------- ML Related ----------------------------------------
void check_predictions_batch(void* buffers[3], void* cl_arg)
{
    size_t const pred_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const pred_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const pred_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Targets vector
    size_t const targ_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const targ_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const targ_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    dahl_fp* correct_predictions = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(pred_nx == targ_nx);
    assert(pred_ny == targ_ny);

    // Loop through each batch
    for (size_t y = 0; y < pred_ny; y++)
    {
        dahl_fp max_val = 0.0F;
        size_t max_index = 0;

        for (size_t x = 0; x < pred_nx; x++)
        {
            dahl_fp current_value = pred[(y * pred_ld) + x];

            if (current_value > max_val)
            {
                max_val = current_value;
                max_index = x;
            }
        }

        *correct_predictions += targ[(y * targ_ld) + max_index] == 1;
    }
}

void cross_entropy_loss_batch(void* buffers[3], void* cl_arg)
{
    // Predictions batch
    size_t const pred_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const pred_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const pred_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Targets batch
    size_t const targ_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const targ_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const targ_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output scalar
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(pred_nx == targ_nx);
    assert(pred_ny == targ_ny);
    assert(pred_ld == targ_ld);

    dahl_fp loss = 0;

    for (size_t y = 0; y < pred_ny; y++)
    {
        for (size_t x = 0; x < pred_nx; x++)
        {
            loss -= (targ[(y*targ_ld)+x] * log(pred[(y*pred_ld)+x]));
        }
    }

    // Divide by batch size and increment into out
    *out += loss / (dahl_fp)pred_ny;
}

void cross_entropy_loss_gradient(void* buffers[3], void* cl_arg)
{
    // Predictions batch
    size_t const pred_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const pred_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const pred_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Targets batch
    size_t const targ_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const targ_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const targ_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output by batch
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(pred_nx == targ_nx);
    assert(pred_ny == targ_ny);
    assert(pred_nx == out_nx);
    assert(pred_ny == out_ny);
    assert(pred_ld == targ_ld);
    assert(pred_ld == out_ld);

    for (size_t y = 0; y < pred_ny; y++)
    {
        for (size_t x = 0; x < pred_nx; x++)
        {
            size_t i = (y * pred_ld) + x;
            out[i] = (pred[i] - targ[i]) / (dahl_fp)pred_ny;
        }
    }
}

void convolution_2d(void* buffers[3], void* cl_arg)
{
    // Input matrix
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Kernel matrix
    size_t const k_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const k_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const k_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const k_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const k_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* kernel = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(out_nx == in_nx - k_nx + 1);
    assert(out_ny == in_ny - k_ny + 1);
    assert(in_nz == k_nz);

    // loop through i,j on axes x,y of the output matrix
    for (size_t j = 0; j < out_ny; j++)
    {
        for (size_t i = 0; i < out_nx; i++)
        {
            dahl_fp cell_res = 0.0F;

            // loop through k,l,m on axes x,y,z of the kernel
            for (size_t m = 0; m < k_nz; m++)
            {
                for (size_t l = 0; l < k_ny; l++)
                {
                    for (size_t k = 0; k < k_nx; k++)
                    {
                        dahl_fp kernel_value = kernel[(m * k_ldz) + (l * k_ldy) + k];
                        // Here we add the offset of the slidding window (i,j) to (k,l)
                        // as they both correspond to (x,y).
                        dahl_fp in_value = in[(m * in_ldz) + ((l + j) * in_ldy) + k + i];
                        
                        cell_res += in_value * kernel_value;
                    }
                }
            }

            out[(j * out_ld) + i] = cell_res;
        }
    }
}
