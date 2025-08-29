#include "../../include/dahl_pooling.h"
#include "../../include/dahl_tasks.h"
#include <stdio.h>
#include <stdlib.h>
#include "../arena/arena.h"
#include "starpu_task.h"

// Not much to init because most of the other fields need to be computed depending on input data.
dahl_pooling* pooling_init(dahl_arena* arena, size_t const pool_size, dahl_shape4d const input_shape)
{
    dahl_pooling* pool = dahl_arena_alloc(arena, sizeof(dahl_pooling));

    pool->pool_size = pool_size;
    pool->input_shape = input_shape;
 
    pool->output_shape = (dahl_shape4d){
        .x = pool->input_shape.x / pool->pool_size, // Image dimensions divided by the pool size
        .y = pool->input_shape.y / pool->pool_size,
        .z = pool->input_shape.z,                   // Channel dimension
        .t = pool->input_shape.t                    // Batch dimension
    };

    pool->mask_batch = tensor_init(arena, pool->input_shape);

    return pool;
}

void _pooling_forward_sample(dahl_block const* input, dahl_block* output, 
                         dahl_block* mask, size_t pool_size)
{
    //partition along channel axis
    block_partition_along_z(input);
    block_partition_along_z_mut(output);
    block_partition_along_z_mut(mask);

    size_t const n_filters = GET_NB_CHILDREN(input);

    for (int c = 0; c < n_filters; c++)
    {
        dahl_matrix const* feature_map = GET_SUB_MATRIX(input, c);
        dahl_matrix* sub_output = GET_SUB_MATRIX_MUT(output, c);
        dahl_matrix* sub_mask = GET_SUB_MATRIX_MUT(mask, c);

        task_matrix_max_pooling(feature_map, sub_output, sub_mask, pool_size);
    }

    block_unpartition(input);
    block_unpartition(output);
    block_unpartition(mask);
}

dahl_tensor* pooling_forward(dahl_arena* arena, dahl_pooling* pool, dahl_tensor const* input_batch)
{
    dahl_tensor* output_batch = tensor_init(arena, pool->output_shape);

    // partition along batch axis
    tensor_partition_along_t(input_batch);
    tensor_partition_along_t_mut(output_batch);
    tensor_partition_along_t_mut(pool->mask_batch);

    size_t const batch_size = GET_NB_CHILDREN(input_batch);

    for (int i = 0; i < batch_size; i++)
    {
        _pooling_forward_sample(
            GET_SUB_BLOCK(input_batch, i),
            GET_SUB_BLOCK_MUT(output_batch, i),
            GET_SUB_BLOCK_MUT(pool->mask_batch, i),
            pool->pool_size
        );
    }

    tensor_unpartition(input_batch);
    tensor_unpartition(output_batch);
    tensor_unpartition(pool->mask_batch);

    return output_batch;
}

void _pooling_backward_sample(dahl_block* dl_dinput, dahl_block const* mask, 
                              dahl_block const* dl_dout, size_t pool_size)
{
    // Partition by channel dimension
    block_partition_along_z_mut(dl_dinput);
    block_partition_along_z(mask);
    block_partition_along_z(dl_dout);

    size_t const n_channels = GET_NB_CHILDREN(mask);

    for (int c = 0; c < n_channels; c++)
    {
        dahl_matrix* sub_dl_dinput = GET_SUB_MATRIX_MUT(dl_dinput, c);
        dahl_matrix const* sub_mask = GET_SUB_MATRIX(mask, c);
        dahl_matrix const* sub_dl_dout = GET_SUB_MATRIX(dl_dout, c);

        // Here the result is stored in the mask buffer
        task_matrix_backward_max_pooling(sub_dl_dout, sub_mask, sub_dl_dinput, pool_size);
    }

    block_unpartition(dl_dinput);
    block_unpartition(mask);
    block_unpartition(dl_dout);
}

dahl_tensor* pooling_backward(dahl_arena* arena, dahl_pooling* pool, dahl_tensor const* dl_dout_batch)
{
    // Initialize the result buffer, which is the derivative of the input we got from the forward pass
    dahl_tensor* dl_dinput_batch = tensor_init(arena, pool->input_shape);

    // Partition by batch dimension
    tensor_partition_along_t_mut(dl_dinput_batch);
    tensor_partition_along_t(pool->mask_batch);
    tensor_partition_along_t(dl_dout_batch);

    size_t const batch_size = GET_NB_CHILDREN(dl_dinput_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        _pooling_backward_sample(
            GET_SUB_BLOCK_MUT(dl_dinput_batch, i),
            GET_SUB_BLOCK(pool->mask_batch, i),
            GET_SUB_BLOCK(dl_dout_batch, i),
            pool->pool_size
        );
    }

    tensor_unpartition(dl_dinput_batch);
    tensor_unpartition(pool->mask_batch);
    tensor_unpartition(dl_dout_batch);

    // Reset the mask for the next forward pass
    TASK_FILL(pool->mask_batch, 0.0F);

    return dl_dinput_batch;
}
