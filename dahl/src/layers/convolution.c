#include "../../include/dahl_convolution.h"
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
    conv->filters = tensor_init_random(arena, filter_shape, -0.1, 0.1);
    conv->biases = vector_init_random(arena, num_filters, -0.1, 0.1); // One bias per feature map
    conv->scratch_arena = scratch_arena;

    return conv;
}

// Apply the forward pass to a sample
void _convolution_forward_sample(dahl_block* output, dahl_block const* input,
                         dahl_tensor const* filters, dahl_vector const* biases)
{
    // Partition by the filter dimension, each iteration will tackle a feature map
    block_partition_along_z_mut(output);

    size_t const n_filters = GET_NB_CHILDREN(filters);

    for (size_t c = 0; c < n_filters; c++)
    {
        dahl_block const* filter = GET_SUB_BLOCK(filters, c);
        dahl_matrix* feature_map = GET_SUB_MATRIX_MUT(output, c);

        // Here the input is always the same because we need to compute as many
        // feature map requested (num_filters) for the current sample.
        task_convolution_2d(input, filter, feature_map);

        // Add bias to the feature map 
        // TODO: This should not be blocking too much, cause biases is readonly, but watch out
        // FIXME: why do we get value without acquiring biases? very dangerous
        // -> either call acquire/release of we create a partition for vectors? honestly feels useless
        TASK_ADD_VALUE_SELF(feature_map, vector_get_value(biases, c));
    }

    block_unpartition(output);
}

dahl_tensor* convolution_forward(dahl_arena* arena, dahl_convolution* conv, dahl_tensor const* input_batch)
{
    // Initialize the result tensor
    dahl_tensor* output_batch = tensor_init(arena, conv->output_shape);
    
    // Partition by batch dimension
    tensor_partition_along_t_mut(output_batch);
    tensor_partition_along_t(input_batch);

    // Partition the filters already here to prevent synchronization in the sample functions
    tensor_partition_along_t(conv->filters);

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
    tensor_unpartition(conv->filters);

    return output_batch;
}

void _convolution_backward_sample(dahl_arena* arena,
                                  dahl_block const* dl_dout, dahl_block const* input, 
                                  dahl_block* dl_dinput_redux, dahl_tensor* dl_dfilters,
                                  size_t filter_size, dahl_tensor const* filters)
{
    // Here we need padding on dl_dout
    size_t padding = (filter_size - 1) * 2;
    dahl_shape3d padding_shape = block_get_shape(dl_dout);
    padding_shape.x += padding;
    padding_shape.y += padding;
    // TODO: here this is not efficient, a better way would be to create "valid" and "same" mode for the cross_correlation
    dahl_block const* dl_dout_padded = task_block_add_padding_init(arena, dl_dout, padding_shape);

    dahl_shape4d filters_shape = tensor_get_shape(filters);

    // Partition by channel dimension
    block_partition_along_z(input);
    block_partition_along_z(dl_dout_padded);
    block_partition_along_z(dl_dout);

    size_t const num_filters = filters_shape.t; // Output channels

    for (int f = 0; f < num_filters; f++)
    {
        dahl_matrix const* dl_dout_filter = GET_SUB_MATRIX(dl_dout, f);
        dahl_block* dl_df_redux = GET_SUB_BLOCK_MUT(dl_dfilters, f);
        // Enable redux for sub blocks of dl_dfilters so that results gets accumulated between each
        // batch.
        block_enable_redux(dl_df_redux);

        task_convolution_2d_backward_filters(input, dl_dout_filter, dl_df_redux);

        // This computation is only required when the conv layer is not the first one in the network
        dahl_matrix const* dl_dout_padded_filter = GET_SUB_MATRIX(dl_dout_padded, f);
        dahl_block const* filter = GET_SUB_BLOCK(filters, f);
        task_convolution_2d_backward_input(dl_dout_padded_filter, filter, dl_dinput_redux);
    }

    block_unpartition(input);
    block_unpartition(dl_dout_padded);
    block_unpartition(dl_dout);
}

// TODO: We should add a parameter that controls wether the output should be computed or not: if the
// layer is the first one we don't need to, otherwise we do.
// OR, simply implement another function that returns void?
// because otherwise we have to return null? idk
dahl_tensor* convolution_backward(dahl_arena* arena, dahl_convolution* conv, 
                                 dahl_tensor const* dl_dout_batch, double const learning_rate,
                                 dahl_tensor const* input_batch)
{
    // Can already compute dl_dbiases by summing over axes (x,y,t) to update the biases.
    dahl_vector* dl_dbiases = task_tensor_sum_xyt_axes_init(conv->scratch_arena, dl_dout_batch);
    TASK_SCAL_SELF(dl_dbiases, learning_rate);
    TASK_SUB_SELF(conv->biases, dl_dbiases);

    // Initialize the result buffer, which is the derivative of the input we got from the forward pass
    dahl_tensor* dl_dinput_batch_redux = tensor_init_redux(arena, conv->input_shape);

    dahl_tensor* dl_dfilters = tensor_init(conv->scratch_arena, conv->filter_shape);

    // Partition by batch dimension
    tensor_partition_along_t(dl_dout_batch);
    tensor_partition_along_t(input_batch);
    tensor_partition_along_t_mut(dl_dinput_batch_redux);

    // Already partition the filters (over feature maps: num_filters) because they will be accessed in every batches
    tensor_partition_along_t(conv->filters);
    tensor_partition_along_t_mut(dl_dfilters);

    size_t const batch_size = GET_NB_CHILDREN(dl_dout_batch); 


    for (size_t i = 0; i < batch_size; i++)
    {
        _convolution_backward_sample(
            conv->scratch_arena,
            GET_SUB_BLOCK(dl_dout_batch, i),
            GET_SUB_BLOCK(input_batch, i),
            GET_SUB_BLOCK_MUT(dl_dinput_batch_redux, i),
            dl_dfilters,
            conv->filter_size, 
            conv->filters);
    }

    tensor_unpartition(dl_dout_batch);
    tensor_unpartition(input_batch);
    tensor_unpartition(dl_dinput_batch_redux);
    tensor_unpartition(conv->filters);
    tensor_unpartition(dl_dfilters);

    TASK_SCAL_SELF(dl_dfilters, learning_rate);
    TASK_SUB_SELF(conv->filters, dl_dfilters);

    return dl_dinput_batch_redux;
}
