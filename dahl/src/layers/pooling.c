#include "../../include/dahl_layers.h"
#include "starpu.h"
#include <stdio.h>
#include <stdlib.h>

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

void _pooling_forward_sample(dahl_block const* input, 
                             dahl_block* mask,
                             dahl_block* output, 
                             size_t pool_size)
{
    //partition along channel axis
    TASK_WAIT(input, 0);
    dahl_block_p* input_p = block_partition_along_z(input, DAHL_READ);
    dahl_block_p* mask_p = block_partition_along_z(mask, DAHL_MUT);
    dahl_block_p* output_p = block_partition_along_z(output, DAHL_MUT);

    size_t const n_filters = GET_NB_CHILDREN(input_p);

    for (int c = 0; c < n_filters; c++)
    {
        dahl_matrix const* feature_map = GET_SUB_MATRIX(input_p, c);
        dahl_matrix* sub_mask = GET_SUB_MATRIX_MUT(mask_p, c);
        dahl_matrix* sub_output = GET_SUB_MATRIX_MUT(output_p, c);

        task_matrix_max_pooling(feature_map, sub_mask, sub_output, pool_size);
    }

    block_unpartition(input_p);
    block_unpartition(mask_p);
    block_unpartition(output_p);
}

dahl_tensor_p* pooling_forward(dahl_arena* arena, dahl_pooling* pool, dahl_tensor_p const* input_batch_p)
{
    // partition along batch axis
    dahl_tensor_p* output_batch_p = tensor_partition_along_t(
            tensor_init(arena, pool->output_shape), 
            DAHL_MUT);

    dahl_tensor_p* mask_batch_p = tensor_partition_along_t(pool->mask_batch, DAHL_MUT);

    size_t const batch_size = GET_NB_CHILDREN(input_batch_p);

    for (int i = 0; i < batch_size; i++)
    {
        _pooling_forward_sample(
            GET_SUB_BLOCK(input_batch_p, i),
            GET_SUB_BLOCK_MUT(mask_batch_p, i),
            GET_SUB_BLOCK_MUT(output_batch_p, i),
            pool->pool_size
        );
    }

    tensor_unpartition(mask_batch_p);

    return output_batch_p;
}

void _pooling_backward_sample(dahl_block* dl_dinput, dahl_block const* mask, 
                              dahl_block const* dl_dout, size_t pool_size)
{
    // Partition by channel dimension
    dahl_block_p* dl_dinput_p = block_partition_along_z(dl_dinput, DAHL_MUT);
    dahl_block_p* mask_p = block_partition_along_z(mask, DAHL_READ);
    dahl_block_p* dl_dout_p = block_partition_along_z(dl_dout, DAHL_READ);

    size_t const n_filters = GET_NB_CHILDREN(mask_p);

    for (int c = 0; c < n_filters; c++)
    {
        dahl_matrix* sub_dl_dinput = GET_SUB_MATRIX_MUT(dl_dinput_p, c);
        dahl_matrix const* sub_mask = GET_SUB_MATRIX(mask_p, c);
        dahl_matrix const* sub_dl_dout = GET_SUB_MATRIX(dl_dout_p, c);

        task_matrix_backward_max_pooling(sub_dl_dout, sub_mask, sub_dl_dinput, pool_size);
    }

    block_unpartition(dl_dinput_p);
    block_unpartition(mask_p);
    block_unpartition(dl_dout_p);
}

dahl_tensor_p* pooling_backward(dahl_arena* arena, dahl_pooling* pool, dahl_tensor_p const* dl_dout_batch_p)
{

    // Initialize the result buffer, which is the derivative of the input we got from the forward 
    // pass and partition by batch dimension
    dahl_tensor_p* dl_dinput_batch_p = tensor_partition_along_t(
            tensor_init(arena, pool->input_shape),
            DAHL_MUT);

    dahl_tensor_p* mask_batch_p = tensor_partition_along_t(pool->mask_batch, DAHL_READ);

    size_t const batch_size = GET_NB_CHILDREN(dl_dinput_batch_p);

    for (size_t i = 0; i < batch_size; i++)
    {
        _pooling_backward_sample(
            GET_SUB_BLOCK_MUT(dl_dinput_batch_p, i),
            GET_SUB_BLOCK(mask_batch_p, i),
            GET_SUB_BLOCK(dl_dout_batch_p, i),
            pool->pool_size
        );
    }

    tensor_unpartition(mask_batch_p);

    return dl_dinput_batch_p;
}
