#include "../../include/dahl_convolution.h"
#include "../arena/arena.h"
#include <stdio.h>

dahl_convolution* convolution_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_shape4d input_shape, size_t filter_size, size_t num_filters)
{
    dahl_shape3d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = num_filters, // Equivalent to the number of channels of an image
    };

    dahl_shape4d output_shape = {
        .x = input_shape.x - filter_size + 1, // The convolution will reduce the size of the images
        .y = input_shape.y - filter_size + 1,
        .z = input_shape.z, // nb channels
        .t = input_shape.t, // batch size
    };

    // Same as output_shape but without the batch size
    dahl_shape3d bias_shape = {
        .x = output_shape.x,
        .y = output_shape.y,
        .z = output_shape.z,
    };

    dahl_convolution* conv = dahl_arena_alloc(arena, sizeof(dahl_convolution));

    conv->input_shape = input_shape;
    conv->num_filters = num_filters;
    conv->filter_size = filter_size;
    conv->filter_shape = filter_shape;
    conv->output_shape = output_shape;
    conv->filters = block_init_random(arena, filter_shape);
    conv->biases = block_init_random(arena, bias_shape);
    conv->scratch_arena = scratch_arena;

    return conv;
}

// Apply the forward pass to a sample
void _convolution_forward_sample(dahl_block* output, dahl_block const* input,
                         dahl_block const* filters, dahl_block const* biases)
{
    // Partition by channel dimension
    block_partition_along_z(input);
    block_partition_along_z(filters);
    block_partition_along_z_mut(output);

    size_t const n_channels = GET_NB_CHILDREN(filters);

    for (size_t c = 0; c < n_channels; c++)
    {
        dahl_matrix const* input_channel = GET_SUB_MATRIX(input, c);
        dahl_matrix const* filters_channel = GET_SUB_MATRIX(filters, c);
        dahl_matrix* output_channel = GET_SUB_MATRIX_MUT(output, c);
        task_matrix_cross_correlation(input_channel, filters_channel, output_channel);
    }

    block_unpartition(input);
    block_unpartition(filters);
    block_unpartition(output);

    // Add biases to the output
    TASK_ADD_SELF(output, biases);
    TASK_RELU_SELF(output);
}

dahl_tensor* convolution_forward(dahl_arena* arena, dahl_convolution* conv, dahl_tensor const* input_batch)
{
    // Initialize the result tensor
    dahl_tensor* output_batch = tensor_init(arena, conv->output_shape);
    
    // Partition by batch dimension
    tensor_partition_along_t_mut(output_batch);
    tensor_partition_along_t(input_batch);
    size_t const batch_size = GET_NB_CHILDREN(input_batch);

    // Apply forward pass to each sample of the batch. TODO: Note that this pattern could be vectorized in the future.
    for (size_t i = 0; i < batch_size; i++)
    {
       _convolution_forward_sample(
           GET_SUB_BLOCK_MUT(output_batch, i), 
           GET_SUB_BLOCK(input_batch, i), 
           conv->filters, conv->biases); 
    }
    
    tensor_unpartition(output_batch);
    tensor_unpartition(input_batch);

    return output_batch;
}

void _convolution_backward_sample(dahl_arena* arena,
                                  dahl_block const* dl_dout, dahl_block const* input, 
                                  dahl_block* dl_dfilters, dahl_block* dl_dinput,
                                  size_t filter_size, dahl_block const* filters)
{
    // Here we need padding on dl_dout
    size_t padding = (filter_size - 1) * 2;
    dahl_shape3d padding_shape = block_get_shape(dl_dout);
    padding_shape.x += padding;
    padding_shape.y += padding;
    dahl_block const* dl_dout_padded = block_add_padding_init(arena, dl_dout, padding_shape);

    // Partition by channel dimension
    block_partition_along_z(input);
    block_partition_along_z(dl_dout_padded);
    block_partition_along_z(dl_dout);
    block_partition_along_z_mut(dl_dinput);
    block_partition_along_z_mut(dl_dfilters);

    size_t const n_channels = GET_NB_CHILDREN(filters);

    for (int c = 0; c < n_channels; c++)
    {
        dahl_matrix const* sub_input = GET_SUB_MATRIX(input, c);
        dahl_matrix const* sub_dl_dout = GET_SUB_MATRIX(dl_dout, c);
        dahl_matrix* sub_dl_dfilters = GET_SUB_MATRIX_MUT(dl_dfilters, c);

        task_matrix_cross_correlation(sub_input, sub_dl_dout, sub_dl_dfilters);

        // Next lines
        // dL_dinput[i] = correlate2d(dL_dout[i],self.filters[i], mode="full")
        dahl_matrix const* sub_filters = GET_SUB_MATRIX(filters, c);
        dahl_matrix const* sub_dl_dout_padded = GET_SUB_MATRIX(dl_dout_padded, c);
        dahl_matrix* sub_dl_dinput = GET_SUB_MATRIX_MUT(dl_dinput, c);

        task_matrix_cross_correlation(sub_dl_dout_padded, sub_filters, sub_dl_dinput);
    }

    block_unpartition(input);
    block_unpartition(dl_dout_padded);
    block_unpartition(dl_dout);
    block_unpartition(dl_dinput);
    block_unpartition(dl_dfilters);
}

dahl_tensor* convolution_backward(dahl_arena* arena, dahl_convolution* conv, 
                                 dahl_tensor const* dl_dout_batch, double const learning_rate,
                                 dahl_tensor const* input_batch)
{
    // Initialize the result buffer, which is the derivative of the input we got from the forward pass
    dahl_tensor* dl_dinput_batch = tensor_init(arena, conv->input_shape);

    dahl_shape4d dl_dfilters_shape = {
        conv->filter_shape.x,
        conv->filter_shape.y,
        conv->filter_shape.z,
        conv->input_shape.t,  // batch size
    };

    dahl_tensor* dl_dfilters_batch = tensor_init(conv->scratch_arena, dl_dfilters_shape);

    // Partition by batch dimension
    tensor_partition_along_t(dl_dout_batch);
    tensor_partition_along_t(input_batch);
    tensor_partition_along_t_mut(dl_dinput_batch);
    tensor_partition_along_t_mut(dl_dfilters_batch);

    // Already partition the filters because they will be accessed in every batches
    block_partition_along_z(conv->filters);

    size_t const batch_size = GET_NB_CHILDREN(dl_dout_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        _convolution_backward_sample(
            conv->scratch_arena,
            GET_SUB_BLOCK(dl_dout_batch, i),
            GET_SUB_BLOCK(input_batch, i),
            GET_SUB_BLOCK_MUT(dl_dfilters_batch, i),
            GET_SUB_BLOCK_MUT(dl_dinput_batch, i),
            conv->filter_size, 
            conv->filters);
    }

    tensor_unpartition(dl_dfilters_batch);
    tensor_unpartition(dl_dout_batch);
    tensor_unpartition(input_batch);
    tensor_unpartition(dl_dinput_batch);
    block_unpartition(conv->filters);
    
    dahl_block* summed_dl_dfilters = task_tensor_sum_t_axis_init(conv->scratch_arena, dl_dfilters_batch);
    dahl_block* summed_dl_dout = task_tensor_sum_t_axis_init(conv->scratch_arena, dl_dout_batch);

    // Updating filters and biases
    // filters -= dl_dfilters * learning_rate
    // biases -= dl_dout * learning_rate
    TASK_SCAL_SELF(summed_dl_dfilters, learning_rate / batch_size);
    TASK_SUB_SELF(conv->filters, summed_dl_dfilters);

    TASK_SCAL_SELF(summed_dl_dout, learning_rate / batch_size);
    TASK_SUB_SELF(conv->biases, summed_dl_dout);

    return dl_dinput_batch;
}
