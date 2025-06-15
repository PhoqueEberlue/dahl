#include "../../include/dahl_convolution.h"
#include "../arena/arena.h"
#include <stdio.h>

dahl_convolution* convolution_init(dahl_shape2d input_shape, size_t filter_size, size_t num_filters)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena* save_arena = context_arena;
    context_arena = default_arena;

    dahl_shape3d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = num_filters,
    };
    
    dahl_block* filters = block_init_random(filter_shape);

    dahl_shape3d output_shape = {
        .x = input_shape.x - filter_size + 1,
        .y = input_shape.y - filter_size + 1,
        .z = num_filters,
    };

    dahl_block* biases = block_init_random(output_shape);

    dahl_block* output = block_init(output_shape);
    dahl_matrix* dl_dinput = matrix_init(input_shape);

    dahl_convolution* conv = dahl_arena_alloc(sizeof(dahl_convolution));

    // Little trick to initialize const fields dynamically
    *(dahl_shape2d*)&conv->input_shape = input_shape;
    *(size_t*)&conv->num_filters = num_filters;
    *(size_t*)&conv->filter_size = filter_size;
    *(dahl_shape3d*)&conv->filter_shape = filter_shape;
    *(dahl_shape3d*)&conv->output_shape = output_shape;
    conv->filters = filters;
    conv->biases = biases;
    conv->output = output;
    conv->dl_dinput = dl_dinput;

    context_arena = save_arena;

    return conv;
}

dahl_block* convolution_forward(dahl_convolution* conv, dahl_matrix const* input)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* save_arena = context_arena;
    context_arena = temporary_arena;
    
    block_partition_along_z(conv->output);
    block_partition_along_z(conv->filters);

    // Every block should have the same number of sub matrices
    size_t const n_channels = block_get_sub_matrix_nb(conv->filters);

    for (int i = 0; i < n_channels; i++)
    {
        dahl_matrix* sub_output = block_get_sub_matrix(conv->output, i);
        dahl_matrix* sub_filters = block_get_sub_matrix(conv->filters, i);

        task_matrix_cross_correlation(input, sub_filters, sub_output);
    }
    
    block_unpartition(conv->output);
    block_unpartition(conv->filters);

    TASK_ADD_SELF(conv->output, conv->biases);
    TASK_RELU_SELF(conv->output);

    dahl_arena_reset(temporary_arena);
    context_arena = save_arena;

    return conv->output;
}

dahl_matrix* convolution_backward(dahl_convolution* conv, dahl_block* dl_dout, double const learning_rate, dahl_matrix const* input)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* save_arena = context_arena;
    context_arena = temporary_arena;

    dahl_shape3d tmp_shape = { .x = conv->input_shape.x, .y = conv->input_shape.y, .z = conv->num_filters };
    dahl_block* dl_dinput_tmp = block_init(tmp_shape);

    dahl_block* dl_dfilters = block_init(conv->filter_shape);

    // Here we need padding on dl_dout
    size_t padding = (conv->filter_size - 1) * 2;
    dahl_shape3d padding_shape = block_get_shape(dl_dout);
    padding_shape.x += padding;
    padding_shape.y += padding;
    dahl_block* dl_dout_padded = block_add_padding_init(dl_dout, padding_shape);

    block_partition_along_z(dl_dout_padded);
    block_partition_along_z(dl_dout);
    block_partition_along_z(dl_dfilters);
    block_partition_along_z(conv->filters);
    block_partition_along_z(dl_dinput_tmp);

    // Every block should have the same number of sub matrices
    size_t sub_matrix_nb = block_get_sub_matrix_nb(conv->filters);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix const* sub_dl_dout = block_get_sub_matrix(dl_dout, i);
        dahl_matrix* sub_dl_dfilters = block_get_sub_matrix(dl_dfilters, i);

        task_matrix_cross_correlation(input, sub_dl_dout, sub_dl_dfilters);

        // Next lines
        // dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")
        dahl_matrix const* sub_filters = block_get_sub_matrix(conv->filters, i);
        dahl_matrix const* sub_dl_dout_padded = block_get_sub_matrix(dl_dout_padded, i);
        dahl_matrix* sub_dl_dinput_tmp = block_get_sub_matrix(dl_dinput_tmp, i);

        task_matrix_cross_correlation(sub_dl_dout_padded, sub_filters, sub_dl_dinput_tmp);
    }

    block_unpartition(dl_dout_padded);
    block_unpartition(dl_dout);
    block_unpartition(dl_dfilters);
    block_unpartition(conv->filters);
    block_unpartition(dl_dinput_tmp);

    // Sum the temporary results that were computed just before
    task_block_sum_z_axis(dl_dinput_tmp, conv->dl_dinput);

    // Updating filters and biases
    // filters -= dl_dfilters * learning_rate
    // biases -= dl_dout * learning_rate
    TASK_SCAL_SELF(dl_dfilters, learning_rate);
    TASK_SCAL_SELF(dl_dout, learning_rate);

    TASK_SUB_SELF(conv->filters, dl_dfilters);
    TASK_SUB_SELF(conv->biases, dl_dout);

    dahl_arena_reset(temporary_arena);
    context_arena = save_arena;

    return conv->dl_dinput;
}
