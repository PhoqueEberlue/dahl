#include "codelets.h"
#include "types.h"
#include <assert.h>

// List of neighbours offsets including current position
#define NB_NEIGHBOURS 9

int16_t NEIGHBOURS[NB_NEIGHBOURS][2] = {
    {-1,-1},
    { 0,-1},
    { 1,-1},
    {-1, 0},
    { 0, 0},
    { 1, 0},
    {-1, 1},
    { 0, 1},
    { 1, 1},
};

void cross_correlation_2d(void *buffers[3], void *cl_arg)
{
    // struct params *params = (struct params *)cl_arg;
    
    // Input matrix (passed as a block of (x, y, 1) shape)
    size_t i_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t i_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t i_ld = STARPU_BLOCK_GET_LDY(buffers[0]);
    dahl_fp* i_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Filter block
    size_t f_nx = STARPU_BLOCK_GET_NX(buffers[1]); // filter size
    size_t f_ny = STARPU_BLOCK_GET_NY(buffers[1]); // filter size
    size_t f_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
    dahl_fp* f_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output block
    size_t o_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t o_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t o_ld = STARPU_BLOCK_GET_LDY(buffers[2]);
    dahl_fp* o_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(o_nx == i_nx - f_nx + 1);
    assert(o_ny == i_ny - f_ny + 1);
     
    // Cross correlation, we start at one because of the padding
    for(size_t y = 1; y < i_ny - 1; y++)
    {
        for(size_t x = 1; x < i_nx - 1; x++)
        {
            dahl_fp cell_res = 0.0F;

            for (size_t n = 0; n < NB_NEIGHBOURS; n++)
            {
                int16_t x_offset = NEIGHBOURS[n][0];
                int16_t y_offset = NEIGHBOURS[n][1];

                dahl_fp i_value = i_p[((y + y_offset) * i_ld) + (x + x_offset)];
                dahl_fp f_value = f_p[((1 + y_offset) * f_ld) + (1 + x_offset)];

                cell_res += i_value * f_value;
            }

            // -1 on x,y because of the padding TODO: Doesn't work with a kernel bigger than 3x3
            o_p[((y - 1) * o_ld) + (x - 1)] = cell_res;
        }
    }
}

void relu(void *buffers[1], void *cl_arg)
{
    size_t nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t nz = STARPU_BLOCK_GET_NZ(buffers[0]);
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
    size_t i_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t i_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t i_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t i_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t i_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp* i_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t o_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t o_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t o_ld = STARPU_BLOCK_GET_LDY(buffers[1]);
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
    dahl_fp factor = *(dahl_fp*)cl_arg;

    size_t nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    dahl_fp* p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    for (int i = 0; i < nx*ny*nz; i++)
    {
        p[i] = p[i] * factor;
    }
}

void sub(void *buffers[2], void *cl_arg)
{
    size_t a_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t a_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t a_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t a_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t a_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp* a_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t b_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t b_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t b_nz = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t b_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t b_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp* b_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

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
