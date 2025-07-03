#include "../../include/dahl_dense.h"
#include "starpu_task.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "../arena/arena.h"

dahl_dense* dense_init(dahl_shape4d const input_shape, size_t const n_classes)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_persistent_arena;

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

    dahl_context_arena = save_arena;

    return dense;
}

dahl_matrix* dense_forward(dahl_dense* dense, dahl_tensor* input)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    dense->input_batch = input;

    dahl_shape3d tmp_shape = { 
        .x = dense->output_shape.x, // number of classes
        .y = dense->input_shape.z,  // number of channels
        .z = dense->input_shape.t,  // number of batches
    };

    dahl_block* tmp = block_init(tmp_shape);

    tensor_partition_along_t(dense->input_batch);
    block_partition_along_z(tmp);
    size_t const batch_size = tensor_get_nb_children(dense->input_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* sub_input = tensor_get_sub_block(dense->input_batch, i);
        dahl_matrix* sub_tmp = block_get_sub_matrix(tmp, i);

        block_partition_along_z_flat(sub_input);
        block_partition_along_z(dense->weights);
        matrix_partition_along_y(sub_tmp);

        size_t const n_channels = block_get_nb_children(sub_input);

        for (size_t j = 0; j < n_channels; j++)
        {
            dahl_vector const* sub_input_flatten = block_get_sub_vector(sub_input, j);
            dahl_matrix const* sub_weights = block_get_sub_matrix(dense->weights, j);
            dahl_vector* sub_tmp_channel = matrix_get_sub_vector(sub_tmp, j);

            task_matrix_vector_product(sub_weights, sub_input_flatten, sub_tmp_channel);
        }

        block_unpartition(sub_input);
        block_unpartition(dense->weights);
        matrix_unpartition(sub_tmp);
    }

    tensor_unpartition(dense->input_batch);
    block_unpartition(tmp);

    // Sum over channel dimension, each batch result is kept separated
    dahl_matrix* partial_res = task_block_sum_y_axis_init(tmp);

    matrix_partition_along_y(partial_res);
    matrix_partition_along_y(dense->output_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_vector* sub_partial_res = matrix_get_sub_vector(partial_res, i);
        dahl_vector* sub_output = matrix_get_sub_vector(dense->output_batch, i);
        TASK_ADD_SELF(sub_partial_res, dense->biases);
        task_vector_softmax(sub_partial_res, sub_output);
    }
    
    matrix_unpartition(partial_res);
    matrix_unpartition(dense->output_batch);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return dense->output_batch;
}

dahl_tensor* dense_backward(dahl_dense* dense, dahl_matrix* dl_dout_batch, dahl_fp const learning_rate)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    // Same shape as dl_dout: n classes by batch size
    dahl_matrix* dl_dy_batch = matrix_init(matrix_get_shape(dl_dout_batch));

    matrix_partition_along_y(dl_dout_batch); // list of gradient for each batch
    matrix_partition_along_y(dense->output_batch); // list of predictions from the last forward pass for each batch
    matrix_partition_along_y(dl_dy_batch); // derivative of the output and the gradients

    for (size_t i = 0; i < matrix_get_nb_children(dl_dout_batch); i++)
    {
        dahl_vector const* sub_dl_dout = matrix_get_sub_vector(dl_dout_batch, i);
        dahl_vector const* sub_output = matrix_get_sub_vector(dense->output_batch, i);
        dahl_vector* sub_dl_dy = matrix_get_sub_vector(dl_dy_batch, i);

        dahl_matrix const* tmp = task_vector_softmax_derivative_init(sub_output);
        task_matrix_vector_product(tmp, sub_dl_dout, sub_dl_dy);
    }

    matrix_unpartition(dl_dout_batch);
    matrix_unpartition(dense->output_batch);
    // unpartition dl_dy later

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
    tensor_partition_along_t(dl_dw_batch);
    tensor_partition_along_t(dense->dl_dinput_batch);

    // Weights doesn't have a batch dim, so we will access them only in the channel loop
    block_partition_along_z(dense->weights);

    // Loop through each batch
    for (size_t i = 0; i < tensor_get_nb_children(dense->input_batch); i++)
    {
        dahl_block* input = tensor_get_sub_block(dense->input_batch, i);
        dahl_block* dl_dw = tensor_get_sub_block(dl_dw_batch, i);
        dahl_block* dl_dinput = tensor_get_sub_block(dense->dl_dinput_batch, i);
        dahl_vector const* sub_dl_dy = matrix_get_sub_vector(dl_dy_batch, i); // dl_dy for this current batch

        // Create a clone of dl_dy as a column matrix
        dahl_matrix const* sub_dl_dy_col = vector_to_column_matrix(sub_dl_dy);

        // Partition by channels
        block_partition_along_z(input);
        block_partition_along_z(dl_dw);
        // Here we partition the output of this function (dl_dinput) into flat vectors, because the output 
        // is stored as multiple matrices, however we need to perform the partial computation on flattened
        // views of the matrices.
        block_partition_along_z_flat(dl_dinput);

        size_t const n_channels = block_get_nb_children(input);

        for (size_t j = 0; j < n_channels; j++)
        {
            dahl_matrix* sub_input = block_get_sub_matrix(input, j);
            task_matrix_to_flat_row(sub_input);

            dahl_matrix* sub_dl_dw = block_get_sub_matrix(dl_dw, j);

            task_matrix_matrix_product(sub_dl_dy_col, sub_input, sub_dl_dw);

            dahl_matrix* sub_weights = block_get_sub_matrix(dense->weights, j);
            dahl_matrix const* sub_weights_t = task_matrix_transpose_init(sub_weights);

            dahl_vector* sub_output = block_get_sub_vector(dl_dinput, j);
            task_matrix_vector_product(sub_weights_t, sub_dl_dy, sub_output);  
        }

        block_unpartition(input);
        block_unpartition(dl_dw);
        block_unpartition(dl_dinput);
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
    dahl_context_arena = save_arena;

    return dense->dl_dinput_batch;
}
