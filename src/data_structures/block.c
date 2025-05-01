#include "../../include/dahl_data.h"
#include "../utils.h"
#include "string.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

dahl_block* block_init(dahl_arena* arena, dahl_shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;

    // Arena returns 0 initialized data, no need to declare the elements of the block
    dahl_fp* data = arena_put(arena, n_elems * sizeof(dahl_fp));
    dahl_block* block = arena_put(arena, sizeof(dahl_block));

    block->data = data;
    block->shape = shape;
    block->ldy = shape.x;
    block->ldz = shape.x * shape.y;
    block->is_partitioned = false;
    block->sub_matrices = nullptr;

    return block;
}

dahl_block* block_init_from(dahl_arena* arena, dahl_shape3d const shape, dahl_fp const* data)
{
    dahl_block* block = block_init(arena, shape);
    memcpy(block->data, data, shape.x * shape.y * shape.z);
    return block;
}

dahl_block* block_init_random(dahl_arena* arena, dahl_shape3d const shape)
{
    dahl_block* block = block_init(arena, shape);

    for (int i = 0; i < shape.x * shape.y * shape.z; i += 1)
    {
        block->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return block;
}

dahl_block* block_clone(dahl_arena* arena, dahl_block const* block)
{
    return block_init_from(arena, block->shape, block->data);
}

dahl_block* block_add_padding_init(dahl_arena* arena, dahl_block const* block, dahl_shape3d const new_shape)
{
    dahl_shape3d shape = block->shape;

    assert(new_shape.x >= shape.x && new_shape.y >= shape.y && new_shape.z >= shape.z);

    size_t diff_z = (new_shape.z - shape.z) / 2;
    size_t diff_y = (new_shape.y - shape.y) / 2;
    size_t diff_x = (new_shape.x - shape.x) / 2;

    dahl_block* new_block = block_init(arena, new_shape);

    for (size_t z = 0; z < shape.z; z++)
    {
        for (size_t y = 0; y < shape.y; y++)
        {
            for (size_t x = 0; x < shape.x; x++)
            {
                dahl_fp value = block->data[(z * block->ldz) + (y * block->ldy) + x];
                new_block->data[((z + diff_z) * new_block->ldz) + ((y + diff_y) * new_block->ldy) + (x + diff_x)] = value;
            }
        }

    }

    return new_block;
}

bool block_equals(dahl_block const* a, dahl_block const* b, bool const rounding)
{
    assert(a->shape.x == b->shape.x 
        && a->shape.y == b->shape.y 
        && a->shape.z == b->shape.z);

    bool res = true;

    for (int i = 0; i < (a->shape.x * a->shape.y * a->shape.z); i++)
    {
        if (a->data[i] != b->data[i])
        {
            res = false;
            break;
        }
    }

    return res;
}

void block_partition_along_z(dahl_arena* arena, dahl_block* block)
{
    block->is_partitioned = true;
    block->nb_sub_matrices = block->shape.z;
    block->sub_matrices = arena_put(arena, block->shape.z * sizeof(dahl_matrix));

    for (int z = 0; z < block->shape.z; z++)
    {
        block->sub_matrices[z].data = &block->data[(z * block->ldz)];
        block->sub_matrices[z].is_sub_block_data = true;
        block->sub_matrices[z].shape = (dahl_shape2d){ .x = block->shape.x, .y = block->shape.y };
        block->sub_matrices[z].ld = block->shape.x;
        block->sub_matrices[z].is_partitioned = false;
        block->sub_matrices[z].sub_vectors = nullptr;
    }
}

void block_unpartition(dahl_block* block)
{
    block->sub_matrices = nullptr;
    block->is_partitioned = false;
}

dahl_matrix* block_get_sub_matrix(dahl_block const* block, const size_t index)
{
    assert(block->is_partitioned 
        && block->sub_matrices != nullptr 
        && index < block->nb_sub_matrices);

    return &block->sub_matrices[index];
}

void block_print(dahl_block const* block)
{
    printf("block=%p nx=%zu ny=%zu nz=%zu ldz=%zu ldy=%ldy\n", 
           block->data, block->shape.x, block->shape.y, block->shape.z, block->ldz, block->ldy);

	for(size_t z = 0; z < block->shape.z; z++)
	{
		for(size_t y = 0; y < block->shape.y; y++)
		{
            printf("%s", space_offset(block->shape.y - y - 1));

			for(size_t x = 0; x < block->shape.x; x++)
			{
				printf("%f ", block->data[(z * block->ldz) + (y * block->ldy) + x]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

// dahl_vector* block_to_vector(dahl_block* block)
// {
//     dahl_fp* data = block_data_acquire(block);
//     dahl_shape3d shape = block_get_shape(block);
// 
//     dahl_vector* res = vector_init_from_ptr(shape.x * shape.y * shape.z, data);
// 
//     block_data_release(block);
//     block_finalize_without_data(block);
// 
//     return res;
// }
