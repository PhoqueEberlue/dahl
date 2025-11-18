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

    dense->weights_shape = weights_shape;

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

    matrix_partition_along_y(input_batch, DAHL_READ);
    matrix_partition_along_y(output_batch, DAHL_MUT);

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

void _dense_backward_sample(dahl_vector const* dl_dout,
                            dahl_vector const* input,
                            dahl_matrix* dl_dw_redux,
                            dahl_vector* dl_dinput,
                            dahl_matrix const* weights)
{
    task_vector_outer_product(input, dl_dout, dl_dw_redux);
    task_vector_matrix_product(dl_dout, weights, dl_dinput);  
}

dahl_matrix* dense_backward(dahl_arena* arena, dahl_dense* dense, dahl_matrix const* dl_dout_batch, 
                            dahl_matrix const* input_batch, dahl_fp const learning_rate)
{
    // Already start summing dl_dout_batch result
    dahl_vector* summed_dl_dout = task_matrix_sum_y_axis_init(dense->scratch_arena, dl_dout_batch);
    // Then apply learning rate
    TASK_SCAL_SELF(summed_dl_dout, learning_rate);

    // Initializing the result buffer, representing the derivative of the forward input
    dahl_matrix* dl_dinput_batch = matrix_init(arena, dense->input_shape);

    // Init redux accumulator for the dl_dw partial results in the batch
    dahl_matrix* dl_dw_redux = matrix_init_redux(dense->scratch_arena, dense->weights_shape);

    // Partition by batch
    matrix_partition_along_y(dl_dout_batch, DAHL_READ);
    matrix_partition_along_y(input_batch, DAHL_READ);
    matrix_partition_along_y(dl_dinput_batch, DAHL_MUT);

    size_t const batch_size = GET_NB_CHILDREN(input_batch);
    
    // Loop through each batch
    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_backward_sample(
            GET_SUB_VECTOR(dl_dout_batch, i),
            GET_SUB_VECTOR(input_batch, i),
            dl_dw_redux,
            GET_SUB_VECTOR_MUT(dl_dinput_batch, i),
            dense->weights
        );
    }

    matrix_unpartition(dl_dout_batch);
    matrix_unpartition(input_batch);
    matrix_unpartition(dl_dinput_batch);
    
    // Updating weights, here no need to divide by batch size because it is already done in dl_out_batch
    TASK_SCAL_SELF(dl_dw_redux, learning_rate);
    TASK_SUB_SELF(dense->weights, dl_dw_redux);

    // Updating biases
    TASK_SUB_SELF(dense->biases, summed_dl_dout);

    return dl_dinput_batch;
}
