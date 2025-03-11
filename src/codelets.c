#include "codelets.h"
#include "types.h"

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
    
    // Input matrix
    size_t i_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t i_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t i_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp* i_p = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Filter block
    size_t f_nx = STARPU_BLOCK_GET_NX(buffers[1]); // filter size
    size_t f_ny = STARPU_BLOCK_GET_NY(buffers[1]); // filter size
    size_t f_nz = STARPU_BLOCK_GET_NZ(buffers[1]); // number of filters
    size_t f_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t f_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp* f_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output block
    size_t o_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t o_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t o_nz = STARPU_BLOCK_GET_NZ(buffers[2]); // number of filters
    size_t o_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t o_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* o_p = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(o_nx == i_nx - f_nx + 1);
    assert(o_ny == i_ny - f_ny + 1);
    assert(o_nz == f_nz); // should be the same number of filters
     
    // The filters are on the z axis of the output block
    for(size_t z = 0; z < o_nz; z++)
	{
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
                    dahl_fp f_value = f_p[(z * f_ldz) + ((1 + y_offset) * f_ldy) + (1 + x_offset)];

                    cell_res += i_value * f_value;
                }

                // -1 on x,y because of the padding TODO: Doesn't work with a kernel bigger than 3x3
				o_p[(z * o_ldz) + ((y - 1) * o_ldy) + (x - 1)] = cell_res;
			}
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
        p[i] = p[i] > 0 ? p[i] : 0;
    }
}
