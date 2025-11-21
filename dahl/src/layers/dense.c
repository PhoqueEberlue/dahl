#include "../../include/dahl_layers.h"
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

dahl_matrix_p* dense_forward(dahl_arena* arena, dahl_dense* dense, dahl_matrix_p const* input_batch_p)
{
    dahl_matrix_p* output_batch_p = matrix_partition_along_y(
            matrix_init(arena, dense->output_shape), 
            DAHL_MUT);

    size_t const batch_size = GET_NB_CHILDREN(input_batch_p);

    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_forward_sample(
            dense->scratch_arena,
            GET_SUB_VECTOR(input_batch_p, i),
            GET_SUB_VECTOR_MUT(output_batch_p, i),
            dense->weights,
            dense->biases
        );
    }

    return output_batch_p;
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

dahl_matrix_p* dense_backward(dahl_arena* arena, dahl_dense* dense, dahl_matrix_p const* dl_dout_batch_p, 
                            dahl_matrix_p const* input_batch_p, dahl_fp const learning_rate)
{ 

    // Init redux accumulator for the dl_dw partial results in the batch
    dahl_matrix* dl_dw_redux = matrix_init_redux(dense->scratch_arena, dense->weights_shape);

    // Initializing the result buffer, representing the derivative of the forward input and
    // partition by batch
    dahl_matrix_p* dl_dinput_batch_p = matrix_partition_along_y(
            matrix_init(arena, dense->input_shape),
            DAHL_MUT);

    size_t const batch_size = GET_NB_CHILDREN(input_batch_p);
    
    // Loop through each batch
    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_backward_sample(
            GET_SUB_VECTOR(dl_dout_batch_p, i),
            GET_SUB_VECTOR(input_batch_p, i),
            dl_dw_redux,
            GET_SUB_VECTOR_MUT(dl_dinput_batch_p, i),
            dense->weights
        );
    }
        
    // Updating weights, here no need to divide by batch size because it is already done in dl_out_batch
    TASK_SCAL_SELF(dl_dw_redux, learning_rate);
    TASK_SUB_SELF(dense->weights, dl_dw_redux);

    // Updating biases
    dahl_matrix* dl_dout_batch = matrix_unpartition(dl_dout_batch_p);
    dahl_vector* summed_dl_dout = task_matrix_sum_y_axis_init(dense->scratch_arena, dl_dout_batch);
    REACTIVATE_PARTITION(dl_dout_batch_p);

    // Then apply learning rate
    TASK_SCAL_SELF(summed_dl_dout, learning_rate);
    TASK_SUB_SELF(dense->biases, summed_dl_dout);

    return dl_dinput_batch_p;
}
