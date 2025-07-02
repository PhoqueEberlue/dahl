#include "../../include/dahl_pooling.h"
#include "../../include/dahl_tasks.h"
#include <stdlib.h>
#include "../arena/arena.h"

// Not much to init because most of the other fields need to be computed depending on input data.
dahl_pooling* pooling_init(size_t const pool_size, dahl_shape4d const input_shape)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_persistent_arena;

    dahl_pooling* pool = dahl_arena_alloc(sizeof(dahl_pooling));

    *(size_t*)&pool->pool_size = pool_size;
    *(dahl_shape4d*)&pool->input_shape = input_shape;

    // Image dimensions divided by the pool size
    *(size_t*)&pool->output_shape.x = pool->input_shape.x / pool->pool_size;
    *(size_t*)&pool->output_shape.y = pool->input_shape.y / pool->pool_size;
    // Channel dimension
    *(size_t*)&pool->output_shape.z = pool->input_shape.z;
    // Batch dimension
    *(size_t*)&pool->output_shape.t = pool->input_shape.t;

    dahl_tensor* output = tensor_init(pool->output_shape);
    dahl_tensor* mask = tensor_init(pool->input_shape);

    pool->output_batch = output;
    pool->mask_batch = mask;

    dahl_context_arena = save_arena;

    return pool;
}

dahl_tensor* pooling_forward(dahl_pooling* pool, dahl_tensor* input_batch)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    // Reset the mask
    TASK_FILL(pool->mask_batch, 0.0F);

    // partition along batch axis
    tensor_partition_along_t(input_batch);
    tensor_partition_along_t(pool->mask_batch);
    tensor_partition_along_t(pool->output_batch);

    size_t const batch_size = tensor_get_nb_children(input_batch);

    for (int i = 0; i < batch_size; i++)
    {
        dahl_block* input = tensor_get_sub_block(input_batch, i);
        dahl_block* mask = tensor_get_sub_block(pool->mask_batch, i);
        dahl_block* output = tensor_get_sub_block(pool->output_batch, i);

        // TODO: vectorize this block?
        //partition along channel axis
        block_partition_along_z(input);
        block_partition_along_z(mask);
        block_partition_along_z(output);

        size_t const n_channels = block_get_nb_children(input);

        for (int j = 0; j < n_channels; j++)
        {
            dahl_matrix const* sub_input = block_get_sub_matrix(input, j);
            dahl_matrix* sub_output = block_get_sub_matrix(output, j);
            dahl_matrix* sub_mask = block_get_sub_matrix(mask, j);

            task_matrix_max_pooling(sub_input, sub_output, sub_mask, pool->pool_size);
        }

        block_unpartition(input);
        block_unpartition(mask);
        block_unpartition(output);
    }

    tensor_unpartition(input_batch);
    tensor_unpartition(pool->output_batch);
    tensor_unpartition(pool->mask_batch);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return pool->output_batch;
}

dahl_tensor* pooling_backward(dahl_pooling* pool, dahl_tensor* dl_dout_batch)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    // Partition by batch dimension
    tensor_partition_along_t(pool->mask_batch);
    tensor_partition_along_t(dl_dout_batch);

    for (size_t i = 0; i < tensor_get_nb_children(pool->mask_batch); i++)
    {
        dahl_block* mask = tensor_get_sub_block(pool->mask_batch, i);
        dahl_block* dl_dout = tensor_get_sub_block(dl_dout_batch, i);
        block_partition_along_z(mask);
        block_partition_along_z(dl_dout);

        // Partition by channel dimension
        for (int j = 0; j < block_get_nb_children(mask); j++)
        {
            dahl_matrix* sub_mask = block_get_sub_matrix(mask, j);
            dahl_matrix* sub_dl_dout = block_get_sub_matrix(dl_dout, j);

            task_matrix_backward_max_pooling_self(sub_dl_dout, sub_mask, pool->pool_size);
        }

        block_unpartition(mask);
        block_unpartition(dl_dout);
    }

    tensor_unpartition(pool->mask_batch);
    tensor_unpartition(dl_dout_batch);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return pool->mask_batch;
}
