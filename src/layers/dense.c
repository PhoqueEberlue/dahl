#include "../../include/dahl_dense.h"
#include "starpu_task.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "../arena/arena.h"

dahl_dense* dense_init(dahl_shape4d const input_shape, size_t const n_classes)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena_set_context(dahl_persistent_arena);

    dahl_dense* dense = dahl_arena_alloc(sizeof(dahl_dense));

    dahl_shape2d output_shape = {
        .x = n_classes,     // number of channels
        .y = input_shape.t, // batch size
    };

    dahl_matrix* output = matrix_init(output_shape);

    dahl_tensor* dl_dinput = tensor_init(input_shape);

    *(dahl_shape4d*)&dense->input_shape = input_shape;
    *(dahl_shape2d*)&dense->output_shape = output_shape;

    dahl_shape3d weights_shape = { 
        .x = input_shape.x * input_shape.y, // size of the flattened image
        .y = n_classes,                     // number of classes
        .z = input_shape.z                  // number of channels
    };

    dense->weights =  block_init_random(weights_shape);
    dense->biases =  vector_init_random(n_classes);
    dense->output_batch = output;
    dense->dl_dinput_batch = dl_dinput;

    dahl_arena_restore_context();

    return dense;
}

void _dense_forward_sample(dahl_block const* input, dahl_vector* output, dahl_matrix* tmp, 
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
    dahl_vector* partial_res = task_matrix_sum_y_axis_init(tmp);
    TASK_ADD_SELF(partial_res, biases);
    task_vector_softmax(partial_res, output);
}

dahl_matrix* dense_forward(dahl_dense* dense, dahl_tensor const* input_batch)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena_set_context(dahl_temporary_arena);

    dense->input_batch = input_batch;

    dahl_shape3d tmp_shape = { 
        .x = dense->output_shape.x, // number of classes
        .y = dense->input_shape.z,  // number of channels
        .z = dense->input_shape.t,  // number of batches
    };

    dahl_block* tmp = block_init(tmp_shape);

    tensor_partition_along_t(dense->input_batch);
    block_partition_along_z_mut(tmp);
    matrix_partition_along_y_mut(dense->output_batch);

    size_t const batch_size = GET_NB_CHILDREN(dense->input_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        _dense_forward_sample(
            GET_SUB_BLOCK(dense->input_batch, i),
            GET_SUB_VECTOR_MUT(dense->output_batch, i),
            GET_SUB_MATRIX_MUT(tmp, i),
            dense->weights,
            dense->biases
        );
    }

    tensor_unpartition(dense->input_batch);
    block_unpartition(tmp);
    matrix_unpartition(dense->output_batch);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_arena_restore_context();

    return dense->output_batch;
}

void _dense_backward_sample(dahl_block const* input, dahl_block* dl_dw, 
                            dahl_block* dl_dinput, dahl_vector const* sub_dl_dy,
                            dahl_block const* weights)
{
    // Create a clone of dl_dy as a column matrix
    dahl_matrix const* sub_dl_dy_col = vector_to_column_matrix(sub_dl_dy);

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

        task_matrix_matrix_product(sub_dl_dy_col, sub_input, sub_dl_dw);

        dahl_matrix const* sub_weights = GET_SUB_MATRIX(weights, j);
        dahl_matrix const* sub_weights_t = task_matrix_transpose_init(sub_weights);

        dahl_vector* sub_dl_dinput = GET_SUB_VECTOR_MUT(dl_dinput, j);
        task_matrix_vector_product(sub_weights_t, sub_dl_dy, sub_dl_dinput);  
    }

    block_unpartition(input);
    block_unpartition(dl_dw);
    block_unpartition(dl_dinput);
}

dahl_tensor* dense_backward(dahl_dense* dense, dahl_matrix const* dl_dout_batch, dahl_fp const learning_rate)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena_set_context(dahl_temporary_arena);

    // Same shape as dl_dout: n classes by batch size
    dahl_matrix* dl_dy_batch = matrix_init(matrix_get_shape(dl_dout_batch));

    matrix_partition_along_y(dl_dout_batch); // list of gradient for each batch
    matrix_partition_along_y(dense->output_batch); // list of predictions from the last forward pass for each batch
    matrix_partition_along_y_mut(dl_dy_batch); // derivative of the output and the gradients

    for (size_t i = 0; i < GET_NB_CHILDREN(dl_dout_batch); i++)
    {
        dahl_vector const* sub_dl_dout = GET_SUB_VECTOR(dl_dout_batch, i);
        dahl_vector const* sub_output = GET_SUB_VECTOR(dense->output_batch, i);
        dahl_vector* sub_dl_dy = GET_SUB_VECTOR_MUT(dl_dy_batch, i);

        dahl_matrix const* tmp = task_vector_softmax_derivative_init(sub_output);
        task_matrix_vector_product(tmp, sub_dl_dout, sub_dl_dy);
    }

    matrix_unpartition(dl_dout_batch);
    matrix_unpartition(dense->output_batch);
    matrix_unpartition(dl_dy_batch);

    dahl_shape4d dl_dw_shape = {
        .x = dense->input_shape.x * dense->input_shape.y, // Image size after flattening
        .y = dense->output_shape.x, // Number of classes 
        .z = dense->input_shape.z,  // Number of channels
        .t = dense->input_shape.t   // Batch size
    };

    dahl_tensor* dl_dw_batch = tensor_init(dl_dw_shape);

    // Reset dl_dinput
    TASK_FILL(dense->dl_dinput_batch, 0);

    // Partition by batch
    tensor_partition_along_t(dense->input_batch);
    tensor_partition_along_t_mut(dl_dw_batch);
    tensor_partition_along_t_mut(dense->dl_dinput_batch);
    matrix_partition_along_y(dl_dy_batch);

    // Weights doesn't have a batch dim, so we will access them only in the channel loop
    block_partition_along_z(dense->weights);

    // Loop through each batch
    for (size_t i = 0; i < GET_NB_CHILDREN(dense->input_batch); i++)
    {
        _dense_backward_sample(
            GET_SUB_BLOCK(dense->input_batch, i),
            GET_SUB_BLOCK_MUT(dl_dw_batch, i),
            GET_SUB_BLOCK_MUT(dense->dl_dinput_batch, i),
            GET_SUB_VECTOR(dl_dy_batch, i),
            dense->weights
        );
    }

    tensor_unpartition(dense->input_batch);
    tensor_unpartition(dense->dl_dinput_batch);
    block_unpartition(dense->weights);
    tensor_unpartition(dl_dw_batch);
    matrix_unpartition(dl_dy_batch); // unpartition dl_dy that was partitionned earlier
    
    dahl_vector* summed_dl_dy = task_matrix_sum_y_axis_init(dl_dy_batch);
    dahl_block* summed_dl_dw = task_tensor_sum_t_axis_init(dl_dw_batch);

    // Updating weights
    TASK_SCAL_SELF(summed_dl_dw, learning_rate / dense->input_shape.t); // dl_dw * lr / batch_size
    TASK_SUB_SELF(dense->weights, summed_dl_dw);

    // Updating biases
    TASK_SCAL_SELF(summed_dl_dy, learning_rate / dense->input_shape.t); // dl_dy * lr / batch_size
    TASK_SUB_SELF(dense->biases, summed_dl_dy);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_arena_restore_context();

    return dense->dl_dinput_batch;
}
