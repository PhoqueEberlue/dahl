#include "data_structures.h"
#include "custom_filters.h"
#include "../utils.h"
#include "sys/types.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void* _block_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    metadata* md = dahl_arena_alloc(
        arena,
        // Metadata struct itself
        sizeof(metadata) + 
        // + a flexible array big enough partition pointers to 
        // store all kinds of block partioning
        (BLOCK_NB_PARTITION_TYPE * sizeof(dahl_partition*)
    ));

    for (size_t i = 0; i < BLOCK_NB_PARTITION_TYPE; i++)
        md->partitions[i] = nullptr;

    md->current_partition = -1;
    md->origin_arena = arena; // Saves where the block have been allocated

    dahl_block* block = dahl_arena_alloc(arena, sizeof(dahl_block));
    block->handle = handle;
    block->data = data;
    block->meta = md;

    return block;
}

dahl_block* block_init(dahl_arena* arena, dahl_shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data = dahl_arena_alloc(arena, n_elems * sizeof(dahl_fp));

    for (size_t i = 0; i < n_elems; i++)
        data[i] = 0.0F;

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

    dahl_arena_attach_handle(arena, handle); 

    return _block_init_from_ptr(arena, handle, data);
}

dahl_block* block_init_from(dahl_arena* arena, dahl_shape3d const shape, dahl_fp const* data)
{
    dahl_block* block = block_init(arena, shape);
    size_t const n_elems = shape.x * shape.y * shape.z;

    for (int i = 0; i < n_elems; i++)
    {
        block->data[i] = data[i];
    }

    return block;
}

dahl_block* block_init_random(dahl_arena* arena, dahl_shape3d const shape)
{
    dahl_block* block = block_init(arena, shape);
    size_t const n_elems = shape.x * shape.y * shape.z;

    for (int i = 0; i < n_elems; i += 1)
    {
        block->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return block;
}

// TODO: convert into a starpu task
dahl_block* block_add_padding_init(dahl_arena* arena, dahl_block const* block, dahl_shape3d const new_shape)
{
    dahl_shape3d shape = block_get_shape(block);

    starpu_data_acquire(block->handle, STARPU_R);
    dahl_fp* data = block->data;

    assert(new_shape.x >= shape.x && new_shape.y >= shape.y && new_shape.z >= shape.z);

    size_t diff_z = (new_shape.z - shape.z) / 2;
    size_t diff_y = (new_shape.y - shape.y) / 2;
    size_t diff_x = (new_shape.x - shape.x) / 2;

    dahl_block* res = block_init(arena, new_shape);
    starpu_data_acquire(res->handle, STARPU_W);
    dahl_fp* res_data = res->data;

    for (size_t z = 0; z < shape.z; z++)
    {
        for (size_t y = 0; y < shape.y; y++)
        {
            for (size_t x = 0; x < shape.x; x++)
            {
                dahl_fp value = data[(z * shape.x * shape.y) + (y * shape.x) + x];
                // FIX PLEASE JUST DO AN ACCESSOR FUNCTION WITH X, Y, Z AS PARAMETERS SO WE CAN IGNORE LD
                res_data[((z + diff_z) * new_shape.x * new_shape.y) + ((y + diff_y) * new_shape.x) + (x + diff_x)] = value;
            }
        }

    }

    starpu_data_release(block->handle);
    starpu_data_release(res->handle);

    return res;
}

dahl_shape3d block_get_shape(dahl_block const* block)
{
    size_t nx = starpu_block_get_nx(block->handle);
	size_t ny = starpu_block_get_ny(block->handle);
	size_t nz = starpu_block_get_nz(block->handle);
    dahl_shape3d res = { .x = nx, .y = ny, .z = nz };

    return res;
}

starpu_data_handle_t _block_get_handle(void const* block)
{
    return ((dahl_block*)block)->handle;
}

dahl_partition* _block_get_current_partition(void const* block)
{
    metadata* m = ((dahl_block const*)block)->meta;
    assert(m-> current_partition >= 0 && 
           m->current_partition < BLOCK_NB_PARTITION_TYPE);

    assert(m->partitions[m->current_partition] != nullptr);
    return m->partitions[m->current_partition];
}

size_t _block_get_nb_elem(void const* block)
{
    dahl_shape3d shape = block_get_shape((dahl_block*)block);
    return shape.x * shape.y * shape.z;
}

dahl_fp const* block_data_acquire(dahl_block const* block)
{
    starpu_data_acquire(block->handle, STARPU_R);
    return block->data;
}

dahl_fp* block_data_acquire_mut(dahl_block* block)
{
    starpu_data_acquire(block->handle, STARPU_RW);
    return block->data;
}

void block_data_release(dahl_block const* block)
{
    starpu_data_release(block->handle);
}

bool block_equals(dahl_block const* a, dahl_block const* b, bool const rounding, u_int8_t const precision)
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
        if (rounding)
        {
            if (fp_round(a->data[i], precision) != fp_round(b->data[i], precision))
            {
                res = false;
                break;
            }
        }
        else 
        {
            if (a->data[i] != b->data[i])
            {
                res = false;
                break;
            }
        }
    }

    starpu_data_release(a->handle);
    starpu_data_release(b->handle);

    return res;
}

void _block_partition_along_z(dahl_block const* block, bool is_mut)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit; 

    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_matrix,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, is_mut, block->handle);
}

void block_partition_along_z_mut(dahl_block* block)
{
    _block_partition_along_z(block, true);
}

void block_partition_along_z(dahl_block const* block)
{
    _block_partition_along_z(block, false);
}

void _block_partition_along_z_flat_matrices(dahl_block const* block, bool is_mut, bool is_row)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z_FLAT_MATRICES;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit;

    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = is_row ? starpu_block_filter_pick_matrix_z_as_row_matrix : starpu_block_filter_pick_matrix_z_as_col_matrix,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_matrix,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, is_mut, block->handle);
}

void block_partition_along_z_flat_matrices_mut(dahl_block* block, bool is_row)
{
    _block_partition_along_z_flat_matrices(block, true, is_row);
}

void block_partition_along_z_flat_matrices(dahl_block const* block, bool is_row)
{
    _block_partition_along_z_flat_matrices(block, false, is_row);
}

void _block_partition_along_z_flat_vectors(dahl_block const* block, bool is_mut)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z_FLAT_VECTORS;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit;

    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z_as_flat_vector,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_vector,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, is_mut, block->handle);
}

void block_partition_along_z_flat_vectors_mut(dahl_block* block)
{
    _block_partition_along_z_flat_vectors(block, true);
}

void block_partition_along_z_flat_vectors(dahl_block const* block)
{
    _block_partition_along_z_flat_vectors(block, false);
}

void _block_partition_flatten_to_vector(dahl_block const* block, bool is_mut)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_FLATTEN_TO_VECTOR;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit;

    // Only one vector here because we flatten the whole block into a vector
    size_t const nparts = 1;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_flatten_to_vector,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_vector,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, is_mut, block->handle);

}

void block_partition_flatten_to_vector_mut(dahl_block* block)
{
    _block_partition_flatten_to_vector(block, true);
}

void block_partition_flatten_to_vector(dahl_block const* block)
{
    _block_partition_flatten_to_vector(block, false);
}

void _block_partition_along_z_batch(dahl_block const* block, size_t batch_size, bool is_mut)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z_BATCH;

    size_t const nparts = block_get_shape(block).z / batch_size;

    dahl_partition* p = block->meta->partitions[t];
    // If the partition already exists AND had the same batch size, no need to create it. 
    // FIX Warning, here the memory is lost if we create many partitions with different batch size
    if (p != nullptr && p->nb_children == nparts)
        goto submit;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = nparts,
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_block,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, is_mut, block->handle);
}

void block_partition_along_z_batch_mut(dahl_block* block, size_t batch_size)
{
    _block_partition_along_z_batch(block, batch_size, true);
}

void block_partition_along_z_batch(dahl_block const* block, size_t batch_size)
{
    _block_partition_along_z_batch(block, batch_size, false);
}

void block_unpartition(dahl_block const* block)
{
    dahl_partition* p = block->meta->partitions[block->meta->current_partition];
    assert(p); // Shouldn't crash, an non-active partition is identified by the if bellow
    assert(block->meta->current_partition >= 0 && 
           block->meta->current_partition < BLOCK_NB_PARTITION_TYPE);

    block->meta->current_partition = -1;
    starpu_data_unpartition_submit(block->handle, p->nb_children,
                                   p->handles, STARPU_MAIN_RAM);
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
            // printf("%s", space_offset(shape.y - y - 1));

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
