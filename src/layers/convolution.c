#include "../../include/dahl_convolution.h"
#include "../arena/arena.h"
#include <stdio.h>

dahl_convolution* convolution_init(dahl_shape3d input_shape, size_t filter_size, size_t num_filters)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_persistent_arena;

    dahl_shape3d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = num_filters,
    };
    
    dahl_block* filters = block_init_random(filter_shape);

    dahl_shape4d output_shape = {
        .x = input_shape.x - filter_size + 1,
        .y = input_shape.y - filter_size + 1,
        .z = num_filters,
        .t = input_shape.z, // batch size
    };

    // Same as output_shape but without the batch size
    dahl_shape3d bias_shape = {
        .x = output_shape.x,
        .y = output_shape.y,
        .z = output_shape.z,
    };

    dahl_block* biases = block_init_random(bias_shape);

    dahl_tensor* output = tensor_init(output_shape);
    dahl_block* dl_dinput = block_init(input_shape);

    dahl_convolution* conv = dahl_arena_alloc(sizeof(dahl_convolution));

    // Little trick to initialize const fields dynamically
    *(dahl_shape3d*)&conv->input_shape = input_shape;
    *(size_t*)&conv->num_filters = num_filters;
    *(size_t*)&conv->filter_size = filter_size;
    *(dahl_shape3d*)&conv->filter_shape = filter_shape;
    *(dahl_shape4d*)&conv->output_shape = output_shape;
    conv->filters = filters;
    conv->biases = biases;
    conv->input_batch = nullptr;
    conv->output_batch = output;
    conv->dl_dinput_batch = dl_dinput;

    dahl_context_arena = save_arena;

    return conv;
}

dahl_tensor* convolution_forward(dahl_convolution* conv, dahl_block* input_batch)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    // Saves the input for backward pass
    conv->input_batch = input_batch;
    
    // Partition by batch dimension
    tensor_partition_along_t(conv->output_batch);
    block_partition_along_z(conv->input_batch);

    // Partition by channel dimension
    block_partition_along_z(conv->filters);

    size_t const batch_size = block_get_nb_children(conv->input_batch);
    size_t const n_channels = block_get_nb_children(conv->filters);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* sub_output = tensor_get_sub_block(conv->output_batch, i);
        block_partition_along_z(sub_output);
        dahl_matrix* sub_input = block_get_sub_matrix(conv->input_batch, i); // take with index i
        
        for (size_t j = 0; j < n_channels; j++)
        {
            dahl_matrix* sub_output_channel = block_get_sub_matrix(sub_output, j);
            dahl_matrix* sub_filters = block_get_sub_matrix(conv->filters, j);

            task_matrix_cross_correlation(sub_input, sub_filters, sub_output_channel);
        }

        block_unpartition(sub_output);

        // Add biases to the output
        TASK_ADD_SELF(sub_output, conv->biases);
    }
    
    tensor_unpartition(conv->output_batch);
    block_unpartition(conv->filters);
    block_unpartition(conv->input_batch); // partition batch

    TASK_RELU_SELF(conv->output_batch);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return conv->output_batch;
}

dahl_block* convolution_backward(dahl_convolution* conv, dahl_tensor* dl_dout_batch, double const learning_rate)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    // Reset dl_dinput_batch. Nedeed because otherwise the previous iter data will collide and
    // produce wrong results (because block_sum_z_axis increments in the ouput buffer). FIX?
    TASK_FILL(conv->dl_dinput_batch, 0);

    dahl_shape4d tmp_shape = { 
        .x = conv->input_shape.x, // Img x dim
        .y = conv->input_shape.y, // Img y dim
        .z = conv->num_filters,   // Channel dim
        .t = conv->input_shape.z  // Batch size
    };

    dahl_tensor* dl_dinput_batch_tmp = tensor_init(tmp_shape);

    dahl_block* dl_dfilters = block_init(conv->filter_shape);

    // Partition by batch dimension
    tensor_partition_along_t(dl_dout_batch);
    tensor_partition_along_t(dl_dinput_batch_tmp);
    block_partition_along_z(conv->input_batch);
    block_partition_along_z(conv->dl_dinput_batch);

    // Partition by channel dimension
    block_partition_along_z(dl_dfilters);
    block_partition_along_z(conv->filters);

    size_t const batch_size = tensor_get_nb_children(dl_dout_batch);
    size_t const n_channels = block_get_nb_children(conv->filters);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* dl_dout = tensor_get_sub_block(dl_dout_batch, i);
        dahl_block* dl_dinput_tmp = tensor_get_sub_block(dl_dinput_batch_tmp, i);
        dahl_matrix* input = block_get_sub_matrix(conv->input_batch, i);
        dahl_matrix* dl_dinput = block_get_sub_matrix(conv->dl_dinput_batch, i);

        // Here we need padding on dl_dout
        size_t padding = (conv->filter_size - 1) * 2;
        dahl_shape3d padding_shape = block_get_shape(dl_dout);
        padding_shape.x += padding;
        padding_shape.y += padding;
        dahl_block* dl_dout_padded = block_add_padding_init(dl_dout, padding_shape);

        // Partition by channel dimension
        block_partition_along_z(dl_dout_padded);
        block_partition_along_z(dl_dout);
        block_partition_along_z(dl_dinput_tmp);

        // Every block should have the same number of sub matrices
        size_t sub_matrix_nb = block_get_nb_children(conv->filters);

        for (int j = 0; j < n_channels; j++)
        {
            dahl_matrix const* sub_dl_dout = block_get_sub_matrix(dl_dout, j);
            dahl_matrix* sub_dl_dfilters = block_get_sub_matrix(dl_dfilters, j);

            task_matrix_cross_correlation(input, sub_dl_dout, sub_dl_dfilters);

            // Next lines
            // dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")
            dahl_matrix const* sub_filters = block_get_sub_matrix(conv->filters, j);
            dahl_matrix const* sub_dl_dout_padded = block_get_sub_matrix(dl_dout_padded, j);
            dahl_matrix* sub_dl_dinput_tmp = block_get_sub_matrix(dl_dinput_tmp, j);

            task_matrix_cross_correlation(sub_dl_dout_padded, sub_filters, sub_dl_dinput_tmp);
        }

        block_unpartition(dl_dout_padded);
        block_unpartition(dl_dout);
        block_unpartition(dl_dinput_tmp); 

        block_get_shape(dl_dinput_tmp);
        matrix_get_shape(dl_dinput);
        // Sum the temporary results that were computed just before
        task_block_sum_z_axis(dl_dinput_tmp, dl_dinput);
    }

    tensor_unpartition(dl_dinput_batch_tmp);
    block_unpartition(conv->input_batch);
    block_unpartition(conv->dl_dinput_batch);
    block_unpartition(dl_dfilters);
    block_unpartition(conv->filters);
    // unpartition dl_dout_batch after

    // Updating filters and biases
    // filters -= dl_dfilters * learning_rate
    // biases -= dl_dout * learning_rate
    TASK_SCAL_SELF(dl_dfilters, learning_rate);

    TASK_SUB_SELF(conv->filters, dl_dfilters);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* dl_dout = tensor_get_sub_block(dl_dout_batch, i);
        TASK_SCAL_SELF(dl_dout, learning_rate);
        TASK_SUB_SELF(conv->biases, dl_dout);
    }

    tensor_unpartition(dl_dout_batch);

    dahl_arena_reset(dahl_temporary_arena);
    dahl_context_arena = save_arena;

    return conv->dl_dinput_batch;
}
