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
    
    // Input matrix; here we take y and z because the input will be cross correlated along x axis of the output
    size_t i_ny = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t i_nz = STARPU_MATRIX_GET_NY(buffers[0]);
    // size_t i_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    fp_dahl* i_p = (fp_dahl*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Filter block
    size_t f_nx = STARPU_BLOCK_GET_NX(buffers[1]); // number of filters
    size_t f_ny = STARPU_BLOCK_GET_NY(buffers[1]); // filter size
    size_t f_nz = STARPU_BLOCK_GET_NZ(buffers[1]); // filter size
    size_t f_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t f_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    fp_dahl* f_p = (fp_dahl*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output block
    size_t o_nx = STARPU_BLOCK_GET_NX(buffers[2]); // number of filters
    size_t o_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t o_nz = STARPU_BLOCK_GET_NZ(buffers[2]);
    size_t o_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t o_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    fp_dahl* o_p = (fp_dahl*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(o_ny == i_ny - f_ny + 1);
    assert(o_nz == i_nz - f_nz + 1);
     
    // The filters are on the X axis of the output block
    for(size_t x = 0; x<o_nx; x++)
	{
        // Cross correlation
		for(size_t y = 0; y<o_ny; y++)
		{
			for(size_t z = 0; z<o_nz; z++)
			{
                fp_dahl cell_res = 0.0F;

                for (size_t c = 0; c < NB_NEIGHBOURS; c++)
                {
                    int16_t y_offset = NEIGHBOURS[c][0];
                    int16_t z_offset = NEIGHBOURS[c][1];

                    fp_dahl i_value = i_p[((z - z_offset) * i_ny) + (y - y_offset)];
                    fp_dahl f_value = f_p[((1 - z_offset) * f_ldz) + ((1 - y_offset) * f_ldy) + x];

                    cell_res += i_value * f_value;
                }

                // TODO: Wouldn't work with a kernel bigger than 3?
				o_p[(z*o_ldz)+(y*o_ldy)+x] = cell_res;
			}
		}
	}
}
