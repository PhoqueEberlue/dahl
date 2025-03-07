#include "utils.h"
#include <stdio.h>

#define DAHL_MAX_RANDOM_VALUES 100

// Initialize a starpu block at 0 and return its handle
starpu_data_handle_t block_init(const shape3d shape)
{
    // -------------- Init filters array and handle (weights) --------------
    size_t n_elems = shape.x * shape.y * shape.z;
    fp_dahl* block = (fp_dahl*)malloc(n_elems * sizeof(fp_dahl));

    for (int i = 0; i < n_elems; i += 1)
    {
        block[i] = 0;
    }

    starpu_data_handle_t handle = nullptr;
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)block,
        shape.x,
        shape.x*shape.y,
        shape.x,
        shape.y,
        shape.z,
        sizeof(fp_dahl)
    );

    return handle;
}

void block_fill_random(starpu_data_handle_t handle)
{
    size_t nx = starpu_block_get_nx(handle);
	size_t ny = starpu_block_get_ny(handle);
	size_t nz = starpu_block_get_nz(handle);

    size_t n_elems = nx*ny*nz;

    fp_dahl* block = (fp_dahl*)starpu_block_get_local_ptr(handle);

	starpu_data_acquire(handle, STARPU_W);

    for (int i = 0; i < n_elems; i += 1)
    {
        block[i] = (fp_dahl)(rand()%DAHL_MAX_RANDOM_VALUES);
    }

	starpu_data_release(handle);
}

char* space_offset(size_t offset)
{
    char* res = malloc((offset + 1) * sizeof(char));

    for (int i = 0; i<offset; i += 1)
    {
        res[i] = ' ';
    }

    res[offset] = '\0';

    return res;
}

void block_print(fp_dahl *block, size_t nx, size_t ny, size_t nz, size_t ldy, size_t ldz, bool pretty)
{
	printf("block=%p nx=%zu ny=%zu nz=%zu ldy=%zu ldz=%zu\n", block, nx, ny, nz, ldy, ldz);
	for(size_t z=0; z<nz; z++)
	{
		for(size_t y=0; y<ny; y++)
		{
            if (pretty)
            {
                printf("%s", space_offset(ny-y-1));
            }

			for(size_t x=0; x<nx; x++)
			{
				printf("%f ", block[(z*ldz)+(y*ldy)+x]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

void block_print_from_handle(starpu_data_handle_t block_handle)
{
	fp_dahl* block = (fp_dahl*)starpu_block_get_local_ptr(block_handle);
	size_t nx = starpu_block_get_nx(block_handle);
	size_t ny = starpu_block_get_ny(block_handle);
	size_t nz = starpu_block_get_nz(block_handle);
	size_t ldy = starpu_block_get_local_ldy(block_handle);
	size_t ldz = starpu_block_get_local_ldz(block_handle);

	starpu_data_acquire(block_handle, STARPU_R);
	block_print(block, nx, ny, nz, ldy, ldz, true);
	starpu_data_release(block_handle);
}

// Initialize a starpu matrix at 0 and return its handle
starpu_data_handle_t matrix_init(const shape2d shape)
{
    // -------------- Init filters array and handle (weights) --------------
    size_t n_elems = shape.x * shape.y;
    fp_dahl* block = (fp_dahl*)malloc(n_elems * sizeof(fp_dahl));

    for (int i = 0; i < n_elems; i += 1)
    {
        block[i] = 0;
    }

    starpu_data_handle_t handle = nullptr;
    starpu_matrix_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)block,
        shape.x, // ld, i.e. number of elements between rows
        shape.x,
        shape.y,
        sizeof(fp_dahl)
    );

    return handle;
}

void matrix_fill_random(starpu_data_handle_t handle)
{
    size_t nx = starpu_matrix_get_nx(handle);
	size_t ny = starpu_matrix_get_ny(handle);

    size_t n_elems = nx*ny;

    fp_dahl* matrix = (fp_dahl*)starpu_matrix_get_local_ptr(handle);

	starpu_data_acquire(handle, STARPU_W);

    for (int i = 0; i < n_elems; i += 1)
    {
        matrix[i] = (fp_dahl)(rand()%DAHL_MAX_RANDOM_VALUES);
    }

	starpu_data_release(handle);
}

void matrix_print(fp_dahl *block, size_t nx, size_t ny, size_t ld)
{
	printf("block=%p nx=%zu ny=%zu ld=%zu\n", block, nx, ny, ld);

    for(size_t y=0; y<ny; y++)
    {
        for(size_t x=0; x<nx; x++)
        {
            printf("%f ", block[(y * ld) + x]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_print_from_handle(starpu_data_handle_t matrix_handle)
{
	fp_dahl* matrix = (fp_dahl*)starpu_matrix_get_local_ptr(matrix_handle);
    size_t nx = starpu_matrix_get_nx(matrix_handle);
	size_t ny = starpu_matrix_get_ny(matrix_handle);
	size_t ld = starpu_matrix_get_local_ld(matrix_handle);

	starpu_data_acquire(matrix_handle, STARPU_R);
	matrix_print(matrix, nx, ny, ld);
	starpu_data_release(matrix_handle);
}
