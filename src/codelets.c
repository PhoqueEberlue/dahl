#include "codelets.h"
#include "starpu_task_util.h"
#include "types.h"
#include <assert.h>
#include <stdlib.h>


void cross_correlation_2d(void *buffers[3], void *cl_arg)
{
    // Input matrix
    size_t const a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const a_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp const* const a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Filter matrix
    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp const* const b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output matrix
    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* const c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(c_nx == a_nx - b_nx + 1);
    assert(c_ny == a_ny - b_ny + 1);

    // loop through i,j on axes x,y of c
    for (size_t j = 0; j < c_ny; j++)
    {
        for (size_t i = 0; i < c_nx; i++)
        {
            dahl_fp cell_res = 0.0F;

            // loop through k,l on axes x,y of b
            for (size_t l = 0; l < b_ny; l++)
            {
                for (size_t k = 0; k < b_nx; k++)
                {
                    dahl_fp a_value = a[((l + j) * a_ld) + k + i];
                    dahl_fp b_value = b[(l * b_ld) + k];
                    
                   cell_res += a_value * b_value;
                }
            }

            c[(j * c_ld) + i] = cell_res;
        }
    }
}

void relu(void *buffers[1], void *cl_arg)
{
    size_t const nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* const p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    for (int i = 0; i < nx*ny*nz; i++)
    {
        if (p[i] < 0.0F)
        {
            p[i] = 0.0F;
        }
    }
}

void sum_z_axis(void *buffers[2], void *cl_arg)
{
    size_t const i_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const i_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const i_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const i_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const i_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* const i_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const o_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const o_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const o_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* const o_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(i_nx == o_nx);
    assert(i_ny == o_ny);

    for (int z = 0; z < i_nz; z++)
    {
        for (int y = 0; y < i_ny; y++)
        {
            for (int x = 0; x < i_nx; x++)
            {
                o_p[(y * o_ld) + x] += i_p[(z * i_ldz) + (y * i_ldy) + x];
            }
        }
    }
}

void scal(void *buffers[1], void *cl_arg)
{
    dahl_fp factor;
    starpu_codelet_unpack_args(cl_arg, &factor);

    size_t const nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* const p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    for (int i = 0; i < nx*ny*nz; i++)
    {
        p[i] = p[i] * factor;
    }
}

void sub(void *buffers[2], void *cl_arg)
{
    size_t const a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* const a_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* const b_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_nz = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const c_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* const c_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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
                dahl_fp value_a = a_p[(z * a_ldz) + (y * a_ldy) + x];
                dahl_fp value_b = b_p[(z * b_ldz) + (y * b_ldy) + x];
                c_p[(z * c_ldz) + (y * c_ldy) + x] = value_a - value_b;
            }
        }
    }
}

void add(void *buffers[2], void *cl_arg)
{
    size_t const a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* const a_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* const b_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_nz = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const c_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const c_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* const c_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

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
                dahl_fp value_a = a_p[(z * a_ldz) + (y * a_ldy) + x];
                dahl_fp value_b = b_p[(z * b_ldz) + (y * b_ldy) + x];
                c_p[(z * c_ldz) + (y * c_ldy) + x] = value_a + value_b;
            }
        }
    }
}
