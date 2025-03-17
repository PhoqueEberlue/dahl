#include "codelets.h"
#include "starpu_task_util.h"
#include "types.h"
#include <assert.h>
#include <stdlib.h>


void cross_correlation_2d(void *buffers[3], void *cl_arg)
{
    // Input matrix
    const size_t a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    const size_t a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    const size_t a_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    const dahl_fp* a = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Filter block
    const size_t b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    const size_t b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    const size_t b_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    const dahl_fp* b = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output block
    const size_t c_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    const size_t c_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    const size_t c_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* c = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(c_nx == a_nx - b_nx + 1);
    assert(c_ny == a_ny - b_ny + 1);

    // loop through i,j on axes x,y of c
    for (size_t j = 0; j < c_nx; j++)
    {
        for (size_t i = 0; i < c_nx; i++)
        {
            dahl_fp cell_res = 0.0F;

            // loop through k,l on axes x,y of b
            for (size_t l = 0; l < b_nx; l++)
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
    const size_t nx = STARPU_BLOCK_GET_NX(buffers[0]);
    const size_t ny = STARPU_BLOCK_GET_NY(buffers[0]);
    const size_t nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

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
    const size_t i_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    const size_t i_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    const size_t i_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    const size_t i_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    const size_t i_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    const dahl_fp* i_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    const size_t o_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    const size_t o_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    const size_t o_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* o_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

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

    const size_t nx = STARPU_BLOCK_GET_NX(buffers[0]);
    const size_t ny = STARPU_BLOCK_GET_NY(buffers[0]);
    const size_t nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    for (int i = 0; i < nx*ny*nz; i++)
    {
        p[i] = p[i] * factor;
    }
}

void sub(void *buffers[2], void *cl_arg)
{
    const size_t a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    const size_t a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    const size_t a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    const size_t a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    const size_t a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp* a_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    const size_t b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    const size_t b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    const size_t b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    const size_t b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    const size_t b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    const dahl_fp* b_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(a_nx == b_nx);
    assert(a_ny == b_ny);
    assert(a_nz == b_nz);
    assert(a_ldy == b_ldy);
    assert(a_ldz == b_ldz);

    for (int z = 0; z < a_nz; z++)
    {
        for (int y = 0; y < a_ny; y++)
        {
            for (int x = 0; x < a_nx; x++)
            {
                a_p[(z * a_ldz) + (y * a_ldy) + x] -= b_p[(z * a_ldz) + (y * a_ldy) + x];
            }
        }
    }
}

void add(void *buffers[2], void *cl_arg)
{
    const size_t a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    const size_t a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    const size_t a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    const size_t a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    const size_t a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp* a_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    const size_t b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    const size_t b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    const size_t b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    const size_t b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    const size_t b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    const dahl_fp* b_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(a_nx == b_nx);
    assert(a_ny == b_ny);
    assert(a_nz == b_nz);
    assert(a_ldy == b_ldy);
    assert(a_ldz == b_ldz);

    for (int z = 0; z < a_nz; z++)
    {
        for (int y = 0; y < a_ny; y++)
        {
            for (int x = 0; x < a_nx; x++)
            {
                a_p[(z * a_ldz) + (y * a_ldy) + x] += b_p[(z * a_ldz) + (y * a_ldy) + x];
            }
        }
    }
}
