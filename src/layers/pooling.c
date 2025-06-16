#include "../../include/dahl_pooling.h"
#include "../../include/dahl_tasks.h"
#include <stdlib.h>
#include "../arena/arena.h"

// Not much to init because most of the other fields need to be computed depending on input data.
dahl_pooling* pooling_init(size_t const pool_size, dahl_shape3d const input_shape)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_persistent_arena;

    dahl_pooling* pool = dahl_arena_alloc(sizeof(dahl_pooling));

    *(size_t*)&pool->pool_size = pool_size;
    *(dahl_shape3d*)&pool->input_shape = input_shape;

    // Image dimensions
    *(size_t*)&pool->output_shape.x = pool->input_shape.x / pool->pool_size;
    *(size_t*)&pool->output_shape.y = pool->input_shape.y / pool->pool_size;
    // Channel dimension
    *(size_t*)&pool->output_shape.z = pool->input_shape.z;

    dahl_block* output = block_init(pool->output_shape);
    dahl_block* mask = block_init(pool->input_shape);

    pool->output = output;
    pool->mask = mask;

    dahl_context_arena = save_arena;

    return pool;
}

dahl_block* pooling_forward(dahl_pooling* pool, dahl_block* input_data)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    // Reset the mask
    task_block_fill(pool->mask, 0.0F);

    block_partition_along_z(input_data);
    block_partition_along_z(pool->output);
    block_partition_along_z(pool->mask);

    size_t sub_matrix_nb = block_get_sub_matrix_nb(input_data);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix const* sub_input = block_get_sub_matrix(input_data, i);
        dahl_matrix* sub_output = block_get_sub_matrix(pool->output, i);
        dahl_matrix* sub_mask = block_get_sub_matrix(pool->mask, i);

        task_matrix_max_pooling(sub_input, sub_output, sub_mask, pool->pool_size);
    }

    block_unpartition(input_data);
    block_unpartition(pool->output);
    block_unpartition(pool->mask);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return pool->output;
}

dahl_block* pooling_backward(dahl_pooling* pool, dahl_block* dl_dout)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    block_partition_along_z(pool->mask);
    block_partition_along_z(dl_dout);

    size_t sub_matrix_nb = block_get_sub_matrix_nb(pool->mask);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix* sub_mask = block_get_sub_matrix(pool->mask, i);
        dahl_matrix* sub_dl_dout = block_get_sub_matrix(dl_dout, i);

        task_matrix_backward_max_pooling_self(sub_dl_dout, sub_mask, pool->pool_size);
    }

    block_unpartition(pool->mask);
    block_unpartition(dl_dout);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return pool->mask;
}
