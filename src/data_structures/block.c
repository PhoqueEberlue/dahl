#include "data_structures.h"
#include "../utils.h"
#include "starpu_data_filters.h"
#include "starpu_data_interfaces.h"

// This function shouldn't be exposed in the header:
// The data parameter is an array that should be allocated before calling the function
// but its memory will be managed by the same structure.
// In other words, only dahl_block functions should call this constructor.
dahl_block* block_init_from_ptr(const dahl_shape3d shape, dahl_fp* data)
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

dahl_block* block_init_from(dahl_shape3d const shape, dahl_fp const* data)
{
    size_t const n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data_copy = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i++)
    {
        data_copy[i] = data[i];
    }

    return block_init_from_ptr(shape, data_copy);
}

dahl_block* block_init_random(dahl_shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return block_init_from_ptr(shape, data);
}

dahl_block* block_init(dahl_shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = 0;
    }

    return block_init_from_ptr(shape, data);
}

dahl_block* block_clone(dahl_block const* block)
{
    dahl_fp* data = block_data_acquire(block);
    dahl_shape3d shape = block_get_shape(block);

    dahl_block* res = block_init_from(shape, data);
    block_data_release(block);

    return res;
}

dahl_block* block_add_padding_init(dahl_block const* block, dahl_shape3d const new_shape)
{
    dahl_fp* old_data = block_data_acquire(block);
    dahl_shape3d shape = block_get_shape(block);

    assert(new_shape.x >= shape.x && new_shape.y >= shape.y && new_shape.z >= shape.z);

    size_t diff_z = (new_shape.z - shape.z) / 2;
    size_t diff_y = (new_shape.y - shape.y) / 2;
    size_t diff_x = (new_shape.x - shape.x) / 2;

    dahl_block* res = block_init(new_shape);
    dahl_fp* data = block_data_acquire(res);

    for (size_t z = 0; z < shape.z; z++)
    {
        for (size_t y = 0; y < shape.y; y++)
        {
            for (size_t x = 0; x < shape.x; x++)
            {
                dahl_fp old_value = old_data[(z * shape.x * shape.y) + (y * shape.x) + x];
                // FIX PLEASE JUST DO AN ACCESSOR FUNCTION WITH X, Y, Z AS PARAMETERS SO WE CAN IGNORE LD
                data[((z + diff_z) * new_shape.x * new_shape.y) + ((y + diff_y) * new_shape.x) + (x + diff_x)] = old_value;
            }
        }

    }

    block_data_release(block);
    block_data_release(res);

    return res;
}

dahl_shape3d block_get_shape(dahl_block const* block)
{
    // TODO: do I need to acquire data? maybe it updates the field, not so sure though because I would have to call the resize functions
    // So I think its fine like this.
    starpu_data_handle_t handle = block->handle;
    size_t nx = starpu_block_get_nx(handle);
	size_t ny = starpu_block_get_ny(handle);
	size_t nz = starpu_block_get_nz(handle);
    dahl_shape3d res = { .x = nx, .y = ny, .z = nz };

    return res;
}

dahl_fp* block_data_acquire(dahl_block const* block)
{
    starpu_data_acquire(block->handle, STARPU_RW);
    return block->data;
}

void block_data_release(dahl_block const* block)
{
    starpu_data_release(block->handle);
}

bool block_equals(dahl_block const* a, dahl_block const* b, bool const rounding)
{
    dahl_shape3d shape_a = block_get_shape(a);
    dahl_shape3d shape_b = block_get_shape(b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y 
        && shape_a.z == shape_b.z);

    starpu_data_acquire(a->handle, STARPU_R);
    starpu_data_acquire(b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < (shape_a.x * shape_a.y * shape_a.z); i++)
    {
        if (a->data[i] != b->data[i])
        {
            res = false;
            break;
        }
    }

    starpu_data_release(a->handle);
    starpu_data_release(b->handle);

    return res;
}

void block_partition_along_z(dahl_block* block)
{
    dahl_shape3d const shape = block_get_shape(block);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z,
		.nchildren = shape.z,
	};

	starpu_data_partition(block->handle, &f);
    
    block->is_partitioned = true;
    block->sub_matrices = malloc(shape.z * sizeof(dahl_matrix));

    for (int i = 0; i < starpu_data_get_nb_children(block->handle); i++)
    {
		starpu_data_handle_t sub_matrix_handle = starpu_data_get_sub_data(block->handle, 1, i);

        //TODO: Check if the pointer is always valid?
        dahl_fp* data = (dahl_fp*)starpu_matrix_get_local_ptr(sub_matrix_handle);

        block->sub_matrices[i].handle = sub_matrix_handle;
        block->sub_matrices[i].data = data;
        block->sub_matrices[i].is_sub_block_data = true;
        block->sub_matrices[i].is_partitioned = false;
    }
}

void block_unpartition(dahl_block* block)
{
    starpu_data_unpartition(block->handle, STARPU_MAIN_RAM);
    free(block->sub_matrices);
    block->sub_matrices = nullptr;
    block->is_partitioned = false;
}

size_t block_get_sub_matrix_nb(dahl_block const* block)
{
    return starpu_data_get_nb_children(block->handle);
}

void block_print(dahl_block const* block)
{
    const dahl_shape3d shape = block_get_shape(block);
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

void block_finalize(dahl_block* block)
{
    assert(!block->is_partitioned);
    starpu_data_unregister(block->handle);
    free(block->data);
    free(block);
}

void block_finalize_without_data(dahl_block* block)
{
    assert(!block->is_partitioned);
    starpu_data_unregister(block->handle);
    free(block);
}

dahl_matrix* block_get_sub_matrix(dahl_block const* block, const size_t index)
{
    assert(block->is_partitioned 
        && block->sub_matrices != nullptr 
        && index < starpu_data_get_nb_children(block->handle));

    return &block->sub_matrices[index];
}

dahl_vector* block_to_vector(dahl_block* block)
{
    dahl_fp* data = block_data_acquire(block);
    dahl_shape3d shape = block_get_shape(block);

    dahl_vector* res = vector_init_from_ptr(shape.x * shape.y * shape.z, data);

    block_data_release(block);
    block_finalize_without_data(block);

    return res;
}
