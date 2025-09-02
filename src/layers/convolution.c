#include "../../include/dahl_convolution.h"
#include "../arena/arena.h"
#include "starpu_task.h"
#include <stdio.h>

dahl_convolution* convolution_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_shape4d input_shape, size_t filter_size, size_t num_filters)
{
    dahl_shape4d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = input_shape.z, // number of channels of the input image
        .t = num_filters, // number of feature maps
    };

    dahl_shape4d output_shape = {
        .x = input_shape.x - filter_size + 1, // The convolution will reduce the size of the images
        .y = input_shape.y - filter_size + 1,
        .z = num_filters, // number of feature maps
        .t = input_shape.t, // batch size
    };

    dahl_convolution* conv = dahl_arena_alloc(arena, sizeof(dahl_convolution));

    conv->input_shape = input_shape;
    conv->num_filters = num_filters;
    conv->filter_size = filter_size;
    conv->filter_shape = filter_shape;
    conv->output_shape = output_shape;
    conv->filters = tensor_init_random(arena, filter_shape);
    conv->biases = vector_init_random(arena, num_filters); // One bias per feature map
    conv->scratch_arena = scratch_arena;

    return conv;
}

// Apply the forward pass to a sample
void _convolution_forward_sample(dahl_block* output, dahl_block const* input,
                         dahl_tensor const* filters, dahl_vector const* biases)
{
    // Partition by the filter dimension, each iteration will tackle a feature map
    block_partition_along_z_mut(output);
    tensor_partition_along_t(filters);

    size_t const n_filters = GET_NB_CHILDREN(filters);

    for (size_t c = 0; c < n_filters; c++)
    {
        dahl_block const* filter = GET_SUB_BLOCK(filters, c);
        dahl_matrix* feature_map = GET_SUB_MATRIX_MUT(output, c);

        // Here the input is always the same because we need to compute as many
        // feature map requested (num_filters) for the current sample.
        task_convolution_2d(input, filter, feature_map);

        // Add bias to the feature map
        // TODO: make this non-blocking
        TASK_ADD_VALUE_SELF(feature_map, vector_get_value(biases, c));
    }

    tensor_unpartition(filters);
    block_unpartition(output);

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

dahl_tensor* _convolution_backward_sample(dahl_arena* arena,
                                  dahl_block const* dl_dout, dahl_block const* input, 
                                  dahl_block* dl_dinput, dahl_tensor* dl_dfilters,
                                  dahl_vector* dl_dbiases,
                                  size_t filter_size, dahl_tensor const* filters)
{
    // Here we need padding on dl_dout
    size_t padding = (filter_size - 1) * 2;
    dahl_shape3d padding_shape = block_get_shape(dl_dout);
    padding_shape.x += padding;
    padding_shape.y += padding;
    dahl_block const* dl_dout_padded = task_block_add_padding_init(arena, dl_dout, padding_shape);

    dahl_shape4d filters_shape = tensor_get_shape(filters);

    // Partition by channel dimension
    block_partition_along_z(input);
    block_partition_along_z(dl_dout_padded);
    block_partition_along_z(dl_dout);
    block_partition_along_z_mut(dl_dinput);

    // Partition by filters dimension
    tensor_partition_along_t_mut(dl_dfilters);

    size_t const num_filters = filters_shape.t; // Output channels
    size_t const num_channel = filters_shape.z; // Input channels

    for (int f = 0; f < num_filters; f++)
    {
        dahl_matrix const* dl_dout_filter = GET_SUB_MATRIX(dl_dout, f);
        dahl_block* dl_df = GET_SUB_BLOCK_MUT(dl_dfilters, f);
        block_partition_along_z_mut(dl_df);

        dahl_matrix const* dl_dout_padded_filter = GET_SUB_MATRIX(dl_dout_padded, f);
        dahl_block const* filter = GET_SUB_BLOCK(filters, f);
        block_partition_along_z(filter);

        for (int c = 0; c < num_channel; c++)
        {
            dahl_matrix const* input_chann = GET_SUB_MATRIX(input, c);
            dahl_matrix* dl_df_chann = GET_SUB_MATRIX_MUT(dl_df, c);

            task_matrix_cross_correlation(input_chann, dl_dout_filter, dl_df_chann);

            dahl_matrix const* filter_chann = GET_SUB_MATRIX(filter, c);
            dahl_matrix* dl_dinput_chann = GET_SUB_MATRIX_MUT(dl_dinput, c);

            task_matrix_cross_correlation(dl_dout_padded_filter, filter_chann, dl_dinput_chann);
        }

        block_unpartition(dl_df);
        block_unpartition(filter);
    }

    // Sum the derivative values of each feature map into the vector `dl_dbiases`
    task_block_sum_xy_axes(dl_dout, dl_dbiases);

    block_unpartition(input);
    block_unpartition(dl_dout_padded);
    block_unpartition(dl_dout);
    block_unpartition(dl_dinput);
    tensor_unpartition(dl_dfilters);

    return dl_dfilters;
}

dahl_tensor* convolution_backward(dahl_arena* arena, dahl_convolution* conv, 
                                 dahl_tensor const* dl_dout_batch, double const learning_rate,
                                 dahl_tensor const* input_batch)
{
    // Initialize the result buffer, which is the derivative of the input we got from the forward pass
    dahl_tensor* dl_dinput_batch = tensor_init(arena, conv->input_shape);

    // Partition by batch dimension
    tensor_partition_along_t(dl_dout_batch);
    tensor_partition_along_t(input_batch);
    tensor_partition_along_t_mut(dl_dinput_batch);

    // Already partition the filters (over feature maps: num_filters) because they will be accessed in every batches
    tensor_partition_along_t(conv->filters);

    size_t const batch_size = GET_NB_CHILDREN(dl_dout_batch);

    dahl_tensor* summed_dl_dfilters = tensor_init(conv->scratch_arena, conv->filter_shape);
    dahl_vector* summed_dl_dbiases = vector_init(conv->scratch_arena, conv->filter_shape.t);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_tensor* dl_dfilters = tensor_init(arena, conv->filter_shape);
        dahl_vector* dl_dbiases = vector_init(arena, conv->filter_shape.t); // num filters

        _convolution_backward_sample(
            conv->scratch_arena,
            GET_SUB_BLOCK(dl_dout_batch, i),
            GET_SUB_BLOCK(input_batch, i),
            GET_SUB_BLOCK_MUT(dl_dinput_batch, i),
            dl_dfilters,
            dl_dbiases,
            conv->filter_size, 
            conv->filters);

        // Accumulate dl_dfilters and dl_dbiases
        TASK_ADD_SELF(summed_dl_dfilters, dl_dfilters);
        TASK_ADD_SELF(summed_dl_dbiases, dl_dbiases);
    }

    tensor_unpartition(dl_dout_batch);
    tensor_unpartition(input_batch);
    tensor_unpartition(dl_dinput_batch);
    tensor_unpartition(conv->filters);

    // Updating filters and biases
    TASK_SCAL_SELF(summed_dl_dfilters, learning_rate / batch_size);
    TASK_SUB_SELF(conv->filters, summed_dl_dfilters);

    TASK_SCAL_SELF(summed_dl_dbiases, learning_rate / batch_size);
    TASK_SUB_SELF(conv->biases, summed_dl_dbiases);

    return dl_dinput_batch;
}
