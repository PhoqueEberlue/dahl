#include "types.h"
#include "starpu_data_filters.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define DAHL_MAX_RANDOM_VALUES 10

// This function shouldn't be exposed in the header:
// The data parameter is an array that should be allocated before calling the function
// but its memory will be managed by the same structure.
// In other words, only dahl_block functions should call this constructor.
dahl_block* block_init_from_ptr(const shape3d shape, dahl_fp* const data)
{
    starpu_data_handle_t handle = nullptr;
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x*shape.y,
        shape.x,
        shape.y,
        shape.z,
        sizeof(dahl_fp)
    );

    dahl_block* block = malloc(sizeof(dahl_block));
    block->handle = handle;
    block->data = data;
    block->sub_matrices = nullptr;
    block->is_partitioned = false;

    return block;
}

dahl_block* block_init_from(shape3d const shape, dahl_fp* const data)
{
    size_t const n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data_copy = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i++)
    {
        data_copy[i] = data[i];
    }

    return block_init_from_ptr(shape, data_copy);
}

dahl_block* block_init_random(shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( rand() % DAHL_MAX_RANDOM_VALUES ) );
    }

    return block_init_from_ptr(shape, data);
}

dahl_block* block_init(shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = 0;
    }

    return block_init_from_ptr(shape, data);
}

shape3d block_get_shape(dahl_block const *const block)
{
    // TODO: do I need to acquire data? maybe it updates the field, not so sure though because I would have to call the resize functions
    // So I think its fine like this.
    starpu_data_handle_t handle = block->handle;
    size_t nx = starpu_block_get_nx(handle);
	size_t ny = starpu_block_get_ny(handle);
	size_t nz = starpu_block_get_nz(handle);
    shape3d res = { .x = nx, .y = ny, .z = nz };

    return res;
}

bool block_equals(dahl_block const* const block_a, dahl_block const* const block_b)
{
    shape3d shape_a = block_get_shape(block_a);
    shape3d shape_b = block_get_shape(block_b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y 
        && shape_a.z == shape_b.z);

    starpu_data_acquire(block_a->handle, STARPU_R);
    starpu_data_acquire(block_b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < (shape_a.x * shape_a.y * shape_a.z); i++)
    {
        if (block_a->data[i] != block_b->data[i])
        {
            res = false;
            continue;
        }
    }

    starpu_data_release(block_a->handle);
    starpu_data_release(block_b->handle);

    return res;
}

void block_print(dahl_block const* const block)
{
    const shape3d shape = block_get_shape(block);
	const size_t ldy = starpu_block_get_local_ldy(block->handle);
	const size_t ldz = starpu_block_get_local_ldz(block->handle);

	starpu_data_acquire(block->handle, STARPU_R);

    printf("block=%p nx=%zu ny=%zu nz=%zu ldy=%zu ldz=%zu\n", block->data, shape.x, shape.y, shape.z, ldy, ldz);

	for(size_t z = 0; z < shape.z; z++)
	{
		for(size_t y = 0; y < shape.y; y++)
		{
            printf("%s", space_offset(shape.y - y - 1));

			for(size_t x = 0; x < shape.x; x++)
			{
				printf("%f ", block->data[(z*ldz)+(y*ldy)+x]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");

	starpu_data_release(block->handle);
}

// We don't have to free block->data because it should be managed by the user
void block_finalize(dahl_block* block)
{
    if (block->is_partitioned)
    {
        // Case where user forgot to unpartition data
        starpu_data_unpartition(block->handle, STARPU_MAIN_RAM);
    }

    starpu_data_unregister(block->handle);
    free(block->data);
    free(block);
}

void block_partition_along_z(dahl_block* const block)
{
    shape3d const shape = block_get_shape(block);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = shape.z,
	};

	starpu_data_partition(block->handle, &f);
    
    block->is_partitioned = true;
    block->sub_matrices = malloc(shape.z * sizeof(dahl_matrix));

    for (int i = 0; i < starpu_data_get_nb_children(block->handle); i++)
    {
		starpu_data_handle_t sub_matrix_handle = starpu_data_get_sub_data(block->handle, 1, i);

        //TODO: Check if the pointer is always valid?
        dahl_fp* data = (dahl_fp*)starpu_block_get_local_ptr(sub_matrix_handle);

        block->sub_matrices[i].handle = sub_matrix_handle;
        block->sub_matrices[i].data = data;
        block->sub_matrices[i].is_sub_block_data = false;
    }
}

void block_unpartition(dahl_block* const block)
{
    starpu_data_unpartition(block->handle, STARPU_MAIN_RAM);
    free(block->sub_matrices);
    block->sub_matrices = nullptr;
    block->is_partitioned = false;
}

size_t block_get_sub_matrix_nb(dahl_block const* const block)
{
    return starpu_data_get_nb_children(block->handle);
}

dahl_matrix* block_get_sub_matrix(dahl_block const* const block, const size_t index)
{
    assert(block->is_partitioned 
        && block->sub_matrices != nullptr 
        && index < starpu_data_get_nb_children(block->handle));

    return &block->sub_matrices[index];
}


// See `block_init_from_ptr` for more information.
dahl_matrix* matrix_init_from_ptr(shape2d const shape, dahl_fp* const data)
{
    starpu_data_handle_t handle = nullptr;

    // Under the hood, dahl_matrix is in fact a starpu_block with only 1 z dimension
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x*shape.y,
        shape.x,
        shape.y,
        1,
        sizeof(dahl_fp)
    );

    dahl_matrix* matrix = malloc(sizeof(dahl_matrix));
    matrix->handle = handle;
    matrix->data = data;
    matrix->is_sub_block_data = false;
    matrix->is_partitioned = false;
    matrix->sub_vectors = nullptr;

    return matrix;
}

dahl_matrix* matrix_init_from(shape2d const shape, dahl_fp* const data)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data_copy = malloc(n_elems * sizeof(dahl_fp));
    
    // TODO: memcpy doesn't work, it's not a big deal but it would be nice to understand why
    // memcpy(data_copy, data, n_elems);

    for (int i = 0; i < n_elems; i++)
    {
        data_copy[i] = data[i];
    }

    return matrix_init_from_ptr(shape, data_copy);
}

dahl_matrix* matrix_init_random(shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( rand() % DAHL_MAX_RANDOM_VALUES ) );
    }

    return matrix_init_from(shape, data);
}

// Initialize a starpu block at 0 and return its handle
dahl_matrix* matrix_init(shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = 0;
    }

    return matrix_init_from_ptr(shape, data);
}

shape2d matrix_get_shape(dahl_matrix const *const matrix)
{
    size_t nx = starpu_block_get_nx(matrix->handle);
    size_t ny = starpu_block_get_ny(matrix->handle);
    
    shape2d res = { .x = nx, .y = ny };
    return res;
}

bool matrix_equals(dahl_matrix const* const matrix_a, dahl_matrix const* const matrix_b)
{
    shape2d const shape_a = matrix_get_shape(matrix_a);
    shape2d const shape_b = matrix_get_shape(matrix_b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y);

    starpu_data_acquire(matrix_a->handle, STARPU_R);
    starpu_data_acquire(matrix_b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < (shape_a.x * shape_a.y); i++)
    {
        if (matrix_a->data[i] != matrix_b->data[i])
        {
            res = false;
            continue;
        }
    }

    starpu_data_release(matrix_a->handle);
    starpu_data_release(matrix_b->handle);

    return res;
}

void matrix_print(dahl_matrix const* const matrix)
{
    const shape2d shape = matrix_get_shape(matrix);

    // block ldy is equal to matrix ld
	size_t ld = starpu_block_get_local_ldy(matrix->handle);

	starpu_data_acquire(matrix->handle, STARPU_R);

    printf("matrix=%p nx=%zu ny=%zu ld=%zu\n", matrix->data, shape.x, shape.y, ld);

    for(size_t y = 0; y < shape.y; y++)
    {
        printf("%s", space_offset(shape.y - y - 1));

        for(size_t x = 0; x < shape.x; x++)
        {
            printf("%f ", matrix->data[(y*ld)+x]);
        }
        printf("\n");
    }
    printf("\n");

	starpu_data_release(matrix->handle);
}

// We don't have to free matrix->data because it should be managed by the user
void matrix_finalize(dahl_matrix* matrix)
{
    if (matrix->is_partitioned)
    {
        // Case where user forgot to unpartition data
        starpu_data_unpartition(matrix->handle, STARPU_MAIN_RAM);
    }

    if (matrix->is_sub_block_data)
    {
        printf("ERROR: matrix_finalize() shouldn't be used on sub block data because it will be freed by block_unpartition().");
        abort();
    }

    starpu_data_unregister(matrix->handle);
    free(matrix->data);
    free(matrix);
}

void matrix_partition_along_y(dahl_matrix* const matrix)
{
    shape2d const shape = matrix_get_shape(matrix);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = shape.y,
	};

	starpu_data_partition(matrix->handle, &f);
    
    matrix->is_partitioned = true;
    matrix->sub_vectors = malloc(shape.y * sizeof(dahl_matrix));

    for (int i = 0; i < starpu_data_get_nb_children(matrix->handle); i++)
    {
		starpu_data_handle_t sub_vector_handle = starpu_data_get_sub_data(matrix->handle, 1, i);

        dahl_fp* data = (dahl_fp*)starpu_matrix_get_local_ptr(sub_vector_handle);

        matrix->sub_vectors[i].handle = sub_vector_handle;
        matrix->sub_vectors[i].data = data;
        matrix->sub_vectors[i].is_sub_matrix_data = false;
    }
}

void matrix_unpartition(dahl_matrix* const matrix)
{
    starpu_data_unpartition(matrix->handle, STARPU_MAIN_RAM);
    free(matrix->sub_vectors);
    matrix->sub_vectors = nullptr;
    matrix->is_partitioned = false;
}

size_t matrix_get_sub_vector_nb(dahl_matrix const* const matrix)
{
    return starpu_data_get_nb_children(matrix->handle);
}

dahl_vector* matrix_get_sub_matrix(dahl_matrix const* const matrix, const size_t index)
{
    assert(matrix->is_partitioned 
        && matrix->sub_vectors != nullptr 
        && index < starpu_data_get_nb_children(matrix->handle));

    return &matrix->sub_vectors[index];
}

// See `block_init_from_ptr` for more information.
dahl_vector* vector_init_from_ptr(size_t const len, dahl_fp* const data)
{
    starpu_data_handle_t handle = nullptr;

    // Under the hood, dahl_vector is in fact a starpu_block with only 1 y and z dimensions
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        len,   // TODO: is it the len?
        len,   // TODO: is it the len?
        len, // nx
        1,   // ny
        1,   // nz
        sizeof(dahl_fp)
    );

    dahl_vector* vector = malloc(sizeof(dahl_vector));
    vector->handle = handle;
    vector->data = data;
    vector->is_sub_matrix_data = false;

    return vector;
}

dahl_vector* vector_init_from(size_t const len, dahl_fp* const data)
{
    dahl_fp* data_copy = malloc(len * sizeof(dahl_fp));
    
    for (int i = 0; i < len; i++)
    {
        data_copy[i] = data[i];
    }

    return vector_init_from_ptr(len, data_copy);
}

dahl_vector* vector_init_random(size_t const len)
{
    dahl_fp* data = malloc(len * sizeof(dahl_fp));

    for (int i = 0; i < len; i += 1)
    {
        data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( rand() % DAHL_MAX_RANDOM_VALUES ) );
    }

    return vector_init_from(len, data);
}

// Initialize a starpu block at 0 and return its handle
dahl_vector* vector_init(size_t const len)
{
    dahl_fp* data = malloc(len * sizeof(dahl_fp));

    for (int i = 0; i < len; i += 1)
    {
        data[i] = 0;
    }

    return vector_init_from_ptr(len, data);
}

size_t vector_get_len(dahl_vector const *const vector)
{
    return starpu_block_get_nx(vector->handle);
}

bool vector_equals(dahl_vector const* const vector_a, dahl_vector const* const vector_b)
{
    size_t const len_a = vector_get_len(vector_a);
    size_t const len_b = vector_get_len(vector_b);

    assert(len_a == len_b
        && len_a == len_b);

    starpu_data_acquire(vector_a->handle, STARPU_R);
    starpu_data_acquire(vector_b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < (len_a * len_b); i++)
    {
        if (vector_a->data[i] != vector_b->data[i])
        {
            res = false;
            continue;
        }
    }

    starpu_data_release(vector_a->handle);
    starpu_data_release(vector_b->handle);

    return res;
}

void vector_print(dahl_vector const* const vector)
{
    const size_t len = vector_get_len(vector);

	starpu_data_acquire(vector->handle, STARPU_R);

    printf("vector=%p nx=%zu\n", vector->data, len);

    for(size_t x = 0; x < len; x++)
    {
        printf("%f ", vector->data[x]);
    }
    printf("\n");

	starpu_data_release(vector->handle);
}

// We don't have to free matrix->data because it should be managed by the user
void vector_finalize(dahl_vector* vector)
{
    if (vector->is_sub_matrix_data)
    {
        printf("ERROR: vector_finalize() shouldn't be used on sub matrix data because it will be freed by matrix_unpartition().");
        abort();
    }

    starpu_data_unregister(vector->handle);
    free(vector->data);
    free(vector);
}

starpu_data_handle_t any_get_handle(dahl_any const any)
{
    switch (any.type)
    {
        case dahl_type_block:
            return any.structure.block->handle;
        case dahl_type_matrix:
            return any.structure.matrix->handle;
        case dahl_type_vector:
            return any.structure.vector->handle;
    }
}

dahl_any any_zeros_like(dahl_any const any)
{
    switch (any.type)
    {
        case dahl_type_block:
            auto b_shape = block_get_shape(any.structure.block);
            dahl_block* b = block_init(b_shape);
            return AS_ANY(b);
        case dahl_type_matrix:
            auto m_shape = matrix_get_shape(any.structure.matrix);
            dahl_matrix* m = matrix_init(m_shape);
            return AS_ANY(m);
        case dahl_type_vector:
            auto v_len = vector_get_len(any.structure.vector);
            dahl_vector* v = vector_init(v_len);
            return AS_ANY(v);
    }
}
