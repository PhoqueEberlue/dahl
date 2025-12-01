#include "../../include/dahl_layers.h"

dahl_relu* relu_init(dahl_arena* arena, dahl_shape4d input_shape)
{
    dahl_relu* relu = dahl_arena_alloc(arena, sizeof(dahl_relu));

    relu->input_shape = input_shape;
    relu->mask_batch = tensor_init(arena, input_shape);
    return relu;
}

void relu_forward(dahl_relu* relu, dahl_tensor_part* input_batch)
{
    tensor_partition_along_t(relu->mask_batch, DAHL_MUT);
    size_t const batch_size = GET_NB_CHILDREN(input_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* input = GET_SUB_BLOCK_MUT(input_batch, i);
        dahl_block* mask = GET_SUB_BLOCK_MUT(relu->mask_batch, i);
        TASK_RELU_SELF(input, mask);
    }
    
    tensor_unpartition(relu->mask_batch);
}

void relu_backward(dahl_relu* relu, dahl_tensor_part* dl_dout_batch)
{
    tensor_partition_along_t(relu->mask_batch, DAHL_MUT);
    size_t const batch_size = GET_NB_CHILDREN(dl_dout_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* dl_dout = GET_SUB_BLOCK_MUT(dl_dout_batch, i);
        dahl_block* mask = GET_SUB_BLOCK_MUT(relu->mask_batch, i);
        // Multiply dl_dout by the mask so we only keep indexes where values were positive in the
        // forward pass.
        TASK_MUL_SELF(dl_dout, mask);
    }
    
    tensor_unpartition(relu->mask_batch);
}
