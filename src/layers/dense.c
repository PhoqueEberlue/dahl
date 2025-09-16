#include "../../include/dahl_dense.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

dahl_dense* dense_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_shape2d const input_shape, size_t const out_features)
{
    dahl_dense* dense = dahl_arena_alloc(arena, sizeof(dahl_dense));

    dahl_shape2d output_shape = {
        .x = out_features,
        .y = input_shape.y, // batch size
    };

    dense->input_shape = input_shape;
    dense->output_shape = output_shape;

    dahl_shape2d weights_shape = {
        .x = input_shape.x,
        .y = out_features,
    };

    dense->weights = matrix_init_random(arena, weights_shape, -0.1, 0.1);
    dense->biases = vector_init_random(arena, out_features, -0.1, 0.1);
    dense->scratch_arena = scratch_arena;

    return dense;
}

void _dense_forward_sample(dahl_arena* arena,
                           dahl_vector const* input, dahl_vector* output,
                           dahl_matrix const* weights, dahl_vector const* biases)
{
    dahl_vector* partial_res = task_matrix_vector_product_init(arena, weights, input);
    TASK_ADD(partial_res, biases, output);
}

dahl_matrix* dense_forward(dahl_arena* arena, dahl_dense* dense, dahl_matrix const* input_batch)
{
    dahl_matrix* output_batch = matrix_init(arena, dense->output_shape);

    matrix_partition_along_y(input_batch);
    matrix_partition_along_y_mut(output_batch);

    size_t const batch_size = GET_NB_CHILDREN(input_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_forward_sample(
            dense->scratch_arena,
            GET_SUB_VECTOR(input_batch, i),
            GET_SUB_VECTOR_MUT(output_batch, i),
            dense->weights,
            dense->biases
        );
    }

    matrix_unpartition(input_batch);
    matrix_unpartition(output_batch);

    return output_batch;
}

void _dense_backward_sample(dahl_arena* arena,
                            dahl_vector const* dl_dout,
                            dahl_vector const* input,
                            dahl_matrix* dl_dw,
                            dahl_vector* dl_dinput,
                            dahl_matrix const* weights)
{
    task_vector_outer_product(input, dl_dout, dl_dw);

    dahl_matrix const* weights_t = task_matrix_transpose_init(arena, weights);
    task_matrix_vector_product(weights_t, dl_dout, dl_dinput);  
}

dahl_matrix* dense_backward(dahl_arena* arena, dahl_dense* dense, dahl_matrix const* dl_dout_batch, 
                            dahl_matrix const* input_batch, dahl_fp const learning_rate)
{
    // Initializing the result buffer, representing the derivative of the forward input
    dahl_matrix* dl_dinput_batch = matrix_init(arena, dense->input_shape);

    dahl_shape3d dl_dw_shape = {
        .x = dense->input_shape.x,  // Input features
        .y = dense->output_shape.x, // Output features 
        .z = dense->input_shape.y   // Batch size
    };

    dahl_block* dl_dw_batch = block_init(dense->scratch_arena, dl_dw_shape);

    // Partition by batch
    matrix_partition_along_y(dl_dout_batch);
    matrix_partition_along_y(input_batch);
    block_partition_along_z_mut(dl_dw_batch);
    matrix_partition_along_y_mut(dl_dinput_batch);

    size_t const batch_size = GET_NB_CHILDREN(input_batch);
    
    // Loop through each batch
    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_backward_sample(
            dense->scratch_arena,
            GET_SUB_VECTOR(dl_dout_batch, i),
            GET_SUB_VECTOR(input_batch, i),
            GET_SUB_MATRIX_MUT(dl_dw_batch, i),
            GET_SUB_VECTOR_MUT(dl_dinput_batch, i),
            dense->weights
        );
    }

    matrix_unpartition(dl_dout_batch);
    matrix_unpartition(input_batch);
    matrix_unpartition(dl_dinput_batch);
    block_unpartition(dl_dw_batch);
    
    dahl_vector* summed_dl_dout = task_matrix_sum_y_axis_init(dense->scratch_arena, dl_dout_batch);
    dahl_matrix* summed_dl_dw = task_block_sum_z_axis_init(dense->scratch_arena, dl_dw_batch);

    // Updating weights, here no need to divide by batch size because it is already done in dl_out_batch
    TASK_SCAL_SELF(summed_dl_dw, learning_rate);
    TASK_SUB_SELF(dense->weights, summed_dl_dw);

    // Updating biases
    TASK_SCAL_SELF(summed_dl_dout, learning_rate);
    TASK_SUB_SELF(dense->biases, summed_dl_dout);

    return dl_dinput_batch;
}
