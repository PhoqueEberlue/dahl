#include "data_structures.h"
#include "../utils.h"
#include "starpu_data.h"
#include "starpu_data_filters.h"
#include "starpu_data_interfaces.h"
#include "starpu_util.h"
#include "sys/types.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

dahl_block* block_init(dahl_shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    dahl_fp* data = dahl_arena_alloc(n_elems * sizeof(dahl_fp));

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

    dahl_arena_attach_handle(handle);

    dahl_block* block = dahl_arena_alloc(sizeof(dahl_block));
    block->handle = handle;
    block->data = data;
    block->partition_type = DAHL_NONE;
    block->partition_level = 0;

    return block;
}

dahl_block* block_init_from(dahl_shape3d const shape, dahl_fp const* data)
{
    dahl_block* block = block_init(shape);
    size_t const n_elems = shape.x * shape.y * shape.z;

    for (int i = 0; i < n_elems; i++)
    {
        block->data[i] = data[i];
    }

    return block;
}

dahl_block* block_init_random(dahl_shape3d const shape)
{
    dahl_block* block = block_init(shape);
    size_t const n_elems = shape.x * shape.y * shape.z;

    for (int i = 0; i < n_elems; i += 1)
    {
        block->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return block;
}

dahl_block* block_clone(dahl_block const* block)
{
    dahl_shape3d shape = block_get_shape(block);

    dahl_fp* data = block_data_acquire(block);
    dahl_block* res = block_init_from(shape, data);
    block_data_release(block);

    return res;
}

dahl_block* block_add_padding_init(dahl_block const* block, dahl_shape3d const new_shape)
{
    dahl_shape3d shape = block_get_shape(block);

    starpu_data_acquire(block->handle, STARPU_R);
    dahl_fp* data = block->data;

    assert(new_shape.x >= shape.x && new_shape.y >= shape.y && new_shape.z >= shape.z);

    size_t diff_z = (new_shape.z - shape.z) / 2;
    size_t diff_y = (new_shape.y - shape.y) / 2;
    size_t diff_x = (new_shape.x - shape.x) / 2;

    dahl_block* res = block_init(new_shape);
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
    starpu_data_acquire(block->handle, STARPU_R);
    size_t nx = starpu_block_get_nx(block->handle);
	size_t ny = starpu_block_get_ny(block->handle);
	size_t nz = starpu_block_get_nz(block->handle);
    dahl_shape3d res = { .x = nx, .y = ny, .z = nz };
    starpu_data_release(block->handle);

    return res;
}

starpu_data_handle_t _block_get_handle(void const* block)
{
    return ((dahl_block*)block)->handle;
}

size_t _block_get_nb_elem(void const* block)
{
    dahl_shape3d shape = block_get_shape((dahl_block*)block);
    return shape.x * shape.y * shape.z;
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

void block_partition_along_z(dahl_block* block)
{
    // The block shouldn't be already partitionned
    assert(block->partition_type == DAHL_NONE);

    dahl_shape3d const shape = block_get_shape(block);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z,
		.nchildren = shape.z,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

	starpu_data_partition(block->handle, &f);
    
    block->partition_type = DAHL_MATRIX;
    block->sub_data.matrices = dahl_arena_alloc(shape.z * sizeof(dahl_matrix));
    // The new level is equal to the father's level + 1
    uint8_t const new_level = block->partition_level + 1;

    for (int i = 0; i < starpu_data_get_nb_children(block->handle); i++)
    {
		starpu_data_handle_t sub_matrix_handle = starpu_data_get_sub_data(block->handle, new_level, i);

        dahl_fp* data = (dahl_fp*)starpu_matrix_get_local_ptr(sub_matrix_handle);
        assert(data);

        block->sub_data.matrices[i].handle = sub_matrix_handle;
        block->sub_data.matrices[i].data = data;
        block->sub_data.matrices[i].partition_level = new_level;
        block->sub_data.matrices[i].partition_type = DAHL_NONE;
    }
}

// Custom filter
static void starpu_block_filter_pick_matrix_z_as_flat_vector(
    void* parent_interface, void* child_interface, 
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
	struct starpu_block_interface* block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_vector_interface* vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;
	
    blocksize = block_parent->ldz;

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nz, "cannot get %u vectors", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nz, "the chosen vector should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;

    vector_child->nx = nx*ny;

	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			vector_child->ptr = block_parent->ptr + offset;
		
		vector_child->dev_handle = block_parent->dev_handle;
		vector_child->offset = block_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_block_filter_pick_vector_child_ops(
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}

void block_partition_along_z_flat(dahl_block* block)
{
    // The block shouldn't be already partitionned
    assert(block->partition_type == DAHL_NONE);

    dahl_shape3d const shape = block_get_shape(block);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z_as_flat_vector,
		.nchildren = shape.z,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

	starpu_data_partition(block->handle, &f);
    
    block->partition_type = DAHL_VECTOR;
    block->sub_data.vectors = dahl_arena_alloc(shape.z * sizeof(dahl_vector));
    // The new level is equal to the father's level + 1
    uint8_t const new_level = block->partition_level + 1;

    for (int i = 0; i < starpu_data_get_nb_children(block->handle); i++)
    {
		starpu_data_handle_t sub_vector_handle = starpu_data_get_sub_data(block->handle, new_level, i);

        dahl_fp* data = (dahl_fp*)starpu_vector_get_local_ptr(sub_vector_handle);
        assert(data);

        block->sub_data.vectors[i].handle = sub_vector_handle;
        block->sub_data.vectors[i].data = data;
        block->sub_data.vectors[i].partition_level = new_level;
    }
}

// Custom filter that picks the block as vector (flattened view of the block)
static void starpu_block_filter_flatten_to_vector(
    void* parent_interface, void* child_interface, 
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter* f,
    unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nparts)
{
	struct starpu_block_interface* block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_vector_interface* vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;
	
    blocksize = block_parent->ldz;

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG((chunk_pos + id) < nz, "the chosen vector should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;

    // We return only one vector, so it's equal to the product of every dimension size
    vector_child->nx = nx*ny*nz;

	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			vector_child->ptr = block_parent->ptr + offset;
		
		vector_child->dev_handle = block_parent->dev_handle;
		vector_child->offset = block_parent->offset + offset;
	}
}

void block_partition_flatten_to_vector(dahl_block* block)
{
    // The block shouldn't be already partitionned
    assert(block->partition_type == DAHL_NONE);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_flatten_to_vector,
		.nchildren = 1,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

	starpu_data_partition(block->handle, &f);
    
    block->partition_type = DAHL_VECTOR;
    // Only one vector here
    block->sub_data.vectors = dahl_arena_alloc(sizeof(dahl_vector));
    // The new level is equal to the father's level + 1
    uint8_t const new_level = block->partition_level + 1;

    assert(starpu_data_get_nb_children(block->handle) == 1);

    starpu_data_handle_t flat_vector_handle = starpu_data_get_sub_data(block->handle, new_level, 0);
    dahl_fp* data = (dahl_fp*)starpu_vector_get_local_ptr(flat_vector_handle);
    assert(data);

    block->sub_data.vectors[0].handle = flat_vector_handle;
    block->sub_data.vectors[0].data = data;
    block->sub_data.vectors[0].partition_level = new_level;
}

void block_partition_along_z_batch(dahl_block* block, size_t batch_size)
{
    // The block shouldn't be already partitionned
    assert(block->partition_type == DAHL_NONE);

    dahl_shape3d const shape = block_get_shape(block);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_block,
		.nchildren = shape.z / batch_size,
	};

	starpu_data_partition(block->handle, &f);
    
    block->partition_type = DAHL_BLOCK;
    block->sub_data.blocks = dahl_arena_alloc(shape.z * sizeof(dahl_block));
    // The new level is equal to the father's level + 1
    uint8_t const new_level = block->partition_level + 1;

    for (int i = 0; i < starpu_data_get_nb_children(block->handle); i++)
    {
		starpu_data_handle_t sub_block_handle = starpu_data_get_sub_data(block->handle, new_level, i);

        dahl_fp* data = (dahl_fp*)starpu_block_get_local_ptr(sub_block_handle);
        assert(data);

        block->sub_data.blocks[i].handle = sub_block_handle;
        block->sub_data.blocks[i].data = data;
        block->sub_data.blocks[i].partition_level = new_level;
        // Children are not yet partitioned
        block->sub_data.blocks[i].partition_type = DAHL_NONE;
    }
}

void block_unpartition(dahl_block* block)
{
    switch (block->partition_type)
    {
        case DAHL_NONE:
            printf("ERROR: Tried calling %s but the object is not partitioned", __func__);
            abort();
            break;
        case DAHL_TENSOR:
            printf("ERROR: got value %i in function %s, however block should only be partioned into block, matrix or vector", 
                   block->partition_type, __func__);
            abort();
            break;
        case DAHL_BLOCK:
            block->sub_data.blocks = nullptr;
            break;
        case DAHL_MATRIX:
            block->sub_data.matrices = nullptr;
            break;
        case DAHL_VECTOR:
            block->sub_data.vectors = nullptr;
            break;
    }

    block->partition_type = DAHL_NONE;
    starpu_data_unpartition(block->handle, STARPU_MAIN_RAM);
}

size_t block_get_nb_children(dahl_block const* block)
{
    return starpu_data_get_nb_children(block->handle);
}

dahl_block* block_get_sub_block(dahl_block const* block, size_t index)
{
    assert(block->partition_type == DAHL_BLOCK 
        && block->sub_data.blocks != nullptr 
        && index < starpu_data_get_nb_children(block->handle));

    return &block->sub_data.blocks[index];
}

dahl_matrix* block_get_sub_matrix(dahl_block const* block, size_t index)
{
    assert(block->partition_type == DAHL_MATRIX 
        && block->sub_data.matrices != nullptr 
        && index < starpu_data_get_nb_children(block->handle));

    return &block->sub_data.matrices[index];
}

dahl_vector* block_get_sub_vector(dahl_block const* block, size_t index)
{
    assert(block->partition_type == DAHL_VECTOR 
        && block->sub_data.vectors != nullptr 
        && index < starpu_data_get_nb_children(block->handle));

    return &block->sub_data.vectors[index];
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
