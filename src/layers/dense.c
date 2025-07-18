#include "../../include/dahl_dense.h"
#include "starpu_task.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "../arena/arena.h"

dahl_dense* dense_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_shape4d const input_shape, size_t const n_classes)
{
    dahl_dense* dense = dahl_arena_alloc(arena, sizeof(dahl_dense));

    dahl_shape2d output_shape = {
        .x = n_classes,     // number of channels
        .y = input_shape.t, // batch size
    };

    dense->input_shape = input_shape;
    dense->output_shape = output_shape;

    dahl_shape3d weights_shape = { 
        .x = input_shape.x * input_shape.y, // size of the flattened image
        .y = n_classes,                     // number of classes
        .z = input_shape.z                  // number of channels
    };

    dense->weights =  block_init_random(arena, weights_shape);
    dense->biases =  vector_init_random(arena, n_classes);
    dense->scratch_arena = scratch_arena;

    return dense;
}

void _dense_forward_sample(dahl_arena* arena,
                           dahl_block const* input, dahl_vector* output, dahl_matrix* tmp, 
                           dahl_block const* weights, dahl_vector const* biases)
{
    block_partition_along_z_flat_vectors(input);
    matrix_partition_along_y_mut(tmp);
    block_partition_along_z(weights);

    size_t const n_channels = GET_NB_CHILDREN(input);

    for (size_t c = 0; c < n_channels; c++)
    {
        dahl_vector const* sub_input_flatten = GET_SUB_VECTOR(input, c);
        dahl_matrix const* sub_weights = GET_SUB_MATRIX(weights, c);
        dahl_vector* sub_tmp = GET_SUB_VECTOR_MUT(tmp, c);

        task_matrix_vector_product(sub_weights, sub_input_flatten, sub_tmp);
    }

    block_unpartition(input);
    block_unpartition(weights);
    matrix_unpartition(tmp);

    // Sum over channel dimension
    dahl_vector* partial_res = task_matrix_sum_y_axis_init(arena, tmp);
    TASK_ADD_SELF(partial_res, biases);
    task_vector_softmax(partial_res, output);
}

dahl_matrix* dense_forward(dahl_arena* arena, dahl_dense* dense, dahl_tensor const* input_batch)
{
    dahl_matrix* output_batch = matrix_init(arena, dense->output_shape);

    dahl_shape3d tmp_shape = { 
        .x = dense->output_shape.x, // number of classes
        .y = dense->input_shape.z,  // number of channels
        .z = dense->input_shape.t,  // number of batches
    };

    dahl_block* tmp = block_init(dense->scratch_arena, tmp_shape);

    tensor_partition_along_t(input_batch);
    block_partition_along_z_mut(tmp);
    matrix_partition_along_y_mut(output_batch);

    size_t const batch_size = GET_NB_CHILDREN(input_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_forward_sample(
            dense->scratch_arena,
            GET_SUB_BLOCK(input_batch, i),
            GET_SUB_VECTOR_MUT(output_batch, i),
            GET_SUB_MATRIX_MUT(tmp, i),
            dense->weights,
            dense->biases
        );
    }

    tensor_unpartition(input_batch);
    block_unpartition(tmp);
    matrix_unpartition(output_batch);

    dahl_arena_reset(dense->scratch_arena);

    return output_batch;
}

void _dense_backward_sample(dahl_arena* arena,
                            dahl_vector const* dl_dout,
                            dahl_vector const* output,
                            dahl_vector* dl_dy,
                            dahl_block const* input, 
                            dahl_block* dl_dw, 
                            dahl_block* dl_dinput,
                            dahl_block const* weights)
{
    // First compute dl_dy
    dahl_matrix const* tmp = task_vector_softmax_derivative_init(arena, output);
    task_matrix_vector_product(tmp, dl_dout, dl_dy);

    // Create a clone of dl_dy as a column matrix
    dahl_matrix const* dl_dy_col = task_vector_to_column_matrix_init(arena, dl_dy);

    // Partition by channels
    block_partition_along_z_flat_matrices(input, true);
    block_partition_along_z_mut(dl_dw);
    // Here we partition the output of this function (dl_dinput) into flat vectors, because the output 
    // is stored as multiple matrices, however we need to perform the partial computation on flattened
    // views of the matrices.
    block_partition_along_z_flat_vectors_mut(dl_dinput);

    size_t const n_channels = GET_NB_CHILDREN(input); 

    for (size_t j = 0; j < n_channels; j++)
    {
        dahl_matrix const* sub_input = GET_SUB_MATRIX(input, j);
        dahl_matrix* sub_dl_dw = GET_SUB_MATRIX_MUT(dl_dw, j);

        task_matrix_matrix_product(dl_dy_col, sub_input, sub_dl_dw);

        dahl_matrix const* sub_weights = GET_SUB_MATRIX(weights, j);
        dahl_matrix const* sub_weights_t = task_matrix_transpose_init(arena, sub_weights);

        dahl_vector* sub_dl_dinput = GET_SUB_VECTOR_MUT(dl_dinput, j);
        task_matrix_vector_product(sub_weights_t, dl_dy, sub_dl_dinput);  
    }

    block_unpartition(input);
    block_unpartition(dl_dw);
    block_unpartition(dl_dinput);
}

dahl_tensor* dense_backward(dahl_arena* arena, dahl_dense* dense, dahl_matrix const* dl_dout_batch, 
                            dahl_tensor const* input_batch, dahl_matrix const* output_batch, dahl_fp const learning_rate)
{
    // Initializing the result buffer, representing the derivative of the forward input
    dahl_tensor* dl_dinput_batch = tensor_init(arena, dense->input_shape);

    // Same shape as dl_dout: n classes by batch size
    dahl_matrix* dl_dy_batch = matrix_init(dense->scratch_arena, matrix_get_shape(dl_dout_batch));

    dahl_shape4d dl_dw_shape = {
        .x = dense->input_shape.x * dense->input_shape.y, // Image size after flattening
        .y = dense->output_shape.x, // Number of classes 
        .z = dense->input_shape.z,  // Number of channels
        .t = dense->input_shape.t   // Batch size
    };

    dahl_tensor* dl_dw_batch = tensor_init(dense->scratch_arena, dl_dw_shape);

    // Partition by batch
    matrix_partition_along_y(dl_dout_batch);
    matrix_partition_along_y(output_batch);
    matrix_partition_along_y_mut(dl_dy_batch); // derivative of the output and the gradients
    tensor_partition_along_t(input_batch);
    tensor_partition_along_t_mut(dl_dw_batch);
    tensor_partition_along_t_mut(dl_dinput_batch);

    // Weights doesn't have a batch dim, so we will access them only in the channel loop
    block_partition_along_z(dense->weights);

    size_t const batch_size = GET_NB_CHILDREN(input_batch);
    
    // Loop through each batch
    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_backward_sample(
            dense->scratch_arena,
            GET_SUB_VECTOR(dl_dout_batch, i),
            GET_SUB_VECTOR(output_batch, i),
            GET_SUB_VECTOR_MUT(dl_dy_batch, i),
            GET_SUB_BLOCK(input_batch, i),
            GET_SUB_BLOCK_MUT(dl_dw_batch, i),
            GET_SUB_BLOCK_MUT(dl_dinput_batch, i),
            dense->weights
        );
    }

    matrix_unpartition(dl_dout_batch);
    matrix_unpartition(output_batch);
    matrix_unpartition(dl_dy_batch);
    tensor_unpartition(input_batch);
    tensor_unpartition(dl_dinput_batch);
    block_unpartition(dense->weights);
    tensor_unpartition(dl_dw_batch);
    
    dahl_vector* summed_dl_dy = task_matrix_sum_y_axis_init(dense->scratch_arena, dl_dy_batch);
    dahl_block* summed_dl_dw = task_tensor_sum_t_axis_init(dense->scratch_arena, dl_dw_batch);

    // Updating weights
    TASK_SCAL_SELF(summed_dl_dw, learning_rate / batch_size); // dl_dw * lr / batch_size
    TASK_SUB_SELF(dense->weights, summed_dl_dw);

    // Updating biases
    TASK_SCAL_SELF(summed_dl_dy, learning_rate / batch_size); // dl_dy * lr / batch_size
    TASK_SUB_SELF(dense->biases, summed_dl_dy);

    dahl_arena_reset(dense->scratch_arena);

    return dl_dinput_batch;
}
