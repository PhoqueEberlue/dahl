#include "utils.h"
#include "starpu_data_interfaces.h"
#include "types.h"
#include <assert.h>
#include <stdio.h>

#define DAHL_MAX_RANDOM_VALUES 10

starpu_data_handle_t block_init_from(const shape3d shape, dahl_fp block[])
{
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
        sizeof(dahl_fp)
    );

    return handle;
}

// Initialize a starpu block at 0 and return its handle
starpu_data_handle_t block_init(const shape3d shape)
{
    // -------------- Init filters array and handle (weights) --------------
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* block = (dahl_fp*)malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        block[i] = 0;
    }

    return block_init_from(shape, block);
}

void block_fill_random(starpu_data_handle_t handle)
{
    size_t nx = starpu_block_get_nx(handle);
	size_t ny = starpu_block_get_ny(handle);
	size_t nz = starpu_block_get_nz(handle);

    size_t n_elems = nx*ny*nz;

    dahl_fp* block = (dahl_fp*)starpu_block_get_local_ptr(handle);

	starpu_data_acquire(handle, STARPU_W);

    for (int i = 0; i < n_elems; i += 1)
    {
        block[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( rand() % DAHL_MAX_RANDOM_VALUES ) );
    }

	starpu_data_release(handle);
}

bool block_equals(starpu_data_handle_t handle_a, starpu_data_handle_t handle_b)
{
    const size_t a_nx = starpu_block_get_nx(handle_a);
	const size_t a_ny = starpu_block_get_ny(handle_a);
	const size_t a_nz = starpu_block_get_nz(handle_a);
    dahl_fp* a = (dahl_fp*)starpu_block_get_local_ptr(handle_a);

    const size_t b_nx = starpu_block_get_nx(handle_b);
	const size_t b_ny = starpu_block_get_ny(handle_b);
	const size_t b_nz = starpu_block_get_nz(handle_b);
    dahl_fp* b = (dahl_fp*)starpu_block_get_local_ptr(handle_b);

    assert(a_nx == b_nx && a_ny == b_ny && a_nz == b_nz);

    for (int i = 0; i < a_nx*a_ny*a_nz; i++)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }

    return true;
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

void block_print(dahl_fp *block, size_t nx, size_t ny, size_t nz, size_t ldy, size_t ldz, bool pretty)
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
	dahl_fp* block = (dahl_fp*)starpu_block_get_local_ptr(block_handle);
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
    dahl_fp* block = (dahl_fp*)malloc(n_elems * sizeof(dahl_fp));

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
        sizeof(dahl_fp)
    );

    return handle;
}

void matrix_fill_random(starpu_data_handle_t handle)
{
    size_t nx = starpu_matrix_get_nx(handle);
	size_t ny = starpu_matrix_get_ny(handle);

    size_t n_elems = nx*ny;

    dahl_fp* matrix = (dahl_fp*)starpu_matrix_get_local_ptr(handle);

	starpu_data_acquire(handle, STARPU_W);

    for (int i = 0; i < n_elems; i += 1)
    {
        matrix[i] = (dahl_fp)(rand()%DAHL_MAX_RANDOM_VALUES);
    }

	starpu_data_release(handle);
}

void matrix_print(dahl_fp *block, size_t nx, size_t ny, size_t ld)
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
	dahl_fp* matrix = (dahl_fp*)starpu_matrix_get_local_ptr(matrix_handle);
    size_t nx = starpu_matrix_get_nx(matrix_handle);
	size_t ny = starpu_matrix_get_ny(matrix_handle);
	size_t ld = starpu_matrix_get_local_ld(matrix_handle);

	starpu_data_acquire(matrix_handle, STARPU_R);
	matrix_print(matrix, nx, ny, ld);
	starpu_data_release(matrix_handle);
}
