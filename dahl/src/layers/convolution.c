#include "../../include/dahl_layers.h"
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
                         dahl_tensor_p const* filters_p, dahl_vector const* biases)
{
    // Partition by the filter dimension, each iteration will tackle a feature map
    dahl_block_p* output_p = block_partition_along_z(output, DAHL_MUT);

    size_t const n_filters = GET_NB_CHILDREN(filters_p);

    for (size_t c = 0; c < n_filters; c++)
    {
        dahl_block const* filter = GET_SUB_BLOCK(filters_p, c);
        dahl_matrix* feature_map = GET_SUB_MATRIX_MUT(output_p, c);

        // Here the input is always the same because we need to compute as many
        // feature map requested (num_filters) for the current sample.
        task_convolution_2d(input, filter, feature_map);

        // Add bias to the feature map 
        // TODO: This should not be blocking too much, cause biases is readonly, but watch out
        // FIXME: why do we get value without acquiring biases? very dangerous
        // -> either call acquire/release of we create a partition for vectors? honestly feels useless
        TASK_ADD_VALUE_SELF(feature_map, vector_get_value(biases, c));
    }

    block_unpartition(output_p);
}

dahl_tensor_p* convolution_forward(dahl_arena* arena, dahl_convolution* conv, dahl_tensor_p const* input_batch_p)
{
    // Partition by batch dimension
    dahl_tensor_p* output_batch_p = tensor_partition_along_t(
            tensor_init(arena, conv->output_shape),
            DAHL_MUT);

    // Partition the filters already here to prevent synchronization in the sample functions
    dahl_tensor_p* filters_p = tensor_partition_along_t(conv->filters, DAHL_READ);

    size_t const batch_size = GET_NB_CHILDREN(input_batch_p);

    // Apply forward pass to each sample of the batch. TODO: Note that this pattern could be vectorized in the future.
    for (size_t i = 0; i < batch_size; i++)
    {
        _convolution_forward_sample(
            GET_SUB_BLOCK_MUT(output_batch_p, i), 
            GET_SUB_BLOCK(input_batch_p, i), 
            filters_p, conv->biases); 
    }

    tensor_unpartition(filters_p);

    return output_batch_p;
}

void _convolution_backward_sample(dahl_block const* dl_dout, dahl_block const* input, 
                                  dahl_block* dl_dinput_redux, dahl_tensor_p* dl_dfilters_p,
                                  dahl_tensor_p const* filters_p, dahl_vector* dl_dbiases_redux,
                                  dahl_fp const learning_rate, bool is_last_sample,
                                  size_t const num_filters)
{
    task_block_sum_xy_axes(dl_dout, dl_dbiases_redux);

    // Partition by channel dimension
    dahl_block_p* input_p = block_partition_along_z(input, DAHL_READ);
    dahl_block_p* dl_dout_p = block_partition_along_z(dl_dout, DAHL_READ);

    
    for (size_t f = 0; f < num_filters; f++)
    {
        dahl_matrix const* dl_dout_filter = GET_SUB_MATRIX(dl_dout_p, f);
        dahl_block* dl_df_redux = GET_SUB_BLOCK_MUT(dl_dfilters_p, f);
        task_convolution_2d_backward_filters(input, dl_dout_filter, dl_df_redux);

        // Only scale on the last sample: (dl_df1 + df_df2...) * lr = dl_df1 * lr + dl_df2 * lr...
        // but dl_dfx results are aggregated from multiple samples for each filter, that's why we
        // only call for the last one.
        if (is_last_sample) TASK_SCAL_SELF(dl_df_redux, learning_rate);

        // This computation is only required when the conv layer is not the first one in the network
        // TODO: See why redux for this task produces weird scheduling
        // dahl_block const* filter = GET_SUB_BLOCK(filters_p, f);
        // task_convolution_2d_backward_input_padding_free(dl_dout_filter, filter, dl_dinput_redux);
    }

    block_unpartition(input_p);
    block_unpartition(dl_dout_p);
}

// TODO: We should add a parameter that controls wether the output should be computed or not: if the
// layer is the first one we don't need to, otherwise we do.
// OR, simply implement another function that returns void?
// because otherwise we have to return null? idk
dahl_tensor_p* convolution_backward(dahl_arena* arena, dahl_convolution* conv, 
                                 dahl_tensor_p const* dl_dout_batch_p, double const learning_rate,
                                 dahl_tensor_p const* input_batch_p)
{
    // dl_dbiases is computed by summing over axes (x,y,t) to update the biases.
    dahl_vector* dl_dbiases_redux = vector_init_redux(arena, 
            vector_get_len(conv->biases));

    dahl_tensor* dl_dfilters = tensor_init(arena, conv->filter_shape);

    // Initialize the result buffer, which is the derivative of the input we got from the forward 
    // pass, and partition along batch dimension
    dahl_tensor_p* dl_dinput_batch_p = tensor_partition_along_t(
            tensor_init_redux(arena, conv->input_shape), // FIX: Should it be init_redux here? 
            DAHL_REDUX);

    // Already partition the filters (over feature maps: num_filters) because they will be accessed in every batches
    dahl_tensor_p* filters_p = tensor_partition_along_t(conv->filters, DAHL_READ);
    dahl_tensor_p* dl_dfilters_p = tensor_partition_along_t(dl_dfilters, DAHL_REDUX);

    size_t const batch_size = GET_NB_CHILDREN(dl_dout_batch_p);  

    for (size_t i = 0; i < batch_size; i++)
    {
        _convolution_backward_sample(
            GET_SUB_BLOCK(dl_dout_batch_p, i),
            GET_SUB_BLOCK(input_batch_p, i),
            GET_SUB_BLOCK_MUT(dl_dinput_batch_p, i),
            dl_dfilters_p,
            filters_p,
            dl_dbiases_redux,
            learning_rate,
            i==batch_size-1,
            conv->filter_shape.t);
    }

    tensor_unpartition(filters_p);
    tensor_unpartition(dl_dfilters_p);

    TASK_SUB_SELF(conv->filters, dl_dfilters);

    TASK_SCAL_SELF(dl_dbiases_redux, learning_rate);
    TASK_SUB_SELF(conv->biases, dl_dbiases_redux);

    return dl_dinput_batch_p;
}
