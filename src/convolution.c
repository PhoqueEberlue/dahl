#include "convolution.h"
#include "types.h"

convolution* convolution_init(shape2d input_shape, size_t filter_size, size_t num_filters)
{
    shape3d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = num_filters,
    };
    
    dahl_block* filters = block_init_random(filter_shape);

    shape3d output_shape = {
        .x = input_shape.x - filter_size + 1,
        .y = input_shape.y - filter_size + 1,
        .z = num_filters,
    };

    dahl_block* biases = block_init_random(output_shape);

    convolution* conv = malloc(sizeof(convolution));

    // Little trick to initialize const fields dynamically
    *(shape2d*)&conv->input_shape = input_shape;
    *(size_t*)&conv->num_filters = num_filters;
    *(size_t*)&conv->filter_size = filter_size;
    *(shape3d*)&conv->filter_shape = filter_shape;
    *(shape3d*)&conv->output_shape = output_shape;
    conv->filters = filters;
    conv->biases = biases;

    return conv;
}

dahl_block* convolution_forward(convolution* const conv, dahl_matrix const* const input)
{
    // TODO: we may free data between each call of this function? (at each epoch?)
    // so maybe store and free the output from the previous call here?
    // or it should be freed in pooling_forward_pass which should be the last one to use it, 
    // but it seems like a bad idea.
    
    dahl_block* output = block_init(conv->output_shape);

    block_partition_along_z(output);
    block_partition_along_z(conv->filters);

    // Every block should have the same number of sub matrices
    size_t sub_matrix_nb = block_get_sub_matrix_nb(conv->filters);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix* sub_output = block_get_sub_matrix(output, i);
        dahl_matrix* sub_filters = block_get_sub_matrix(conv->filters, i);

        task_matrix_cross_correlation(input, sub_filters, sub_output);
    }
    
    block_unpartition(output);
    block_unpartition(conv->filters);

    task_block_relu(output);

    // Could be interesting to know if the relu task is really waiting for other tasks because starting?
    // It should be the case because of the data dependency and because it is working but we may verify that
    starpu_task_wait_for_all();

    return output;
}

dahl_matrix* convolution_backward(convolution* const conv, dahl_block* const dl_dout, double const learning_rate, dahl_matrix const* const input)
{
    // derivative loss
    dahl_matrix* dl_dinput = matrix_init(conv->input_shape);
    dahl_block* dl_dfilters = block_init(conv->output_shape);

    block_partition_along_z(dl_dout);
    block_partition_along_z(dl_dfilters);
    block_partition_along_z(conv->filters);

    // Every block should have the same number of sub matrices
    size_t sub_matrix_nb = block_get_sub_matrix_nb(conv->filters);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix const* const sub_dl_dout = block_get_sub_matrix(dl_dout, i);
        dahl_matrix* sub_dl_dfilters = block_get_sub_matrix(dl_dfilters, i);

        task_matrix_cross_correlation(input, sub_dl_dout, sub_dl_dfilters);

        // Next four lines
        // dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")
        dahl_matrix const* const sub_filters = block_get_sub_matrix(conv->filters, i);
        dahl_matrix* tmp = matrix_init(conv->input_shape);
        
        task_matrix_cross_correlation(sub_dl_dout, sub_filters, tmp);
        task_matrix_add_self(dl_dinput, tmp);
    }

    starpu_task_wait_for_all();

    block_unpartition(dl_dout);
    block_unpartition(dl_dfilters);
    block_unpartition(conv->filters);

    // Updating filters and biases
    // filters -= dl_dfilters * learning_rate
    // biases -= dl_dout * learning_rate
    task_block_scal_self(dl_dfilters, learning_rate);
    task_block_scal_self(dl_dout, learning_rate);
    task_block_sub_self(conv->filters, dl_dfilters);
    task_block_sub_self(conv->biases, dl_dout);

    starpu_task_wait_for_all();

    block_free(dl_dfilters);

    return dl_dinput;
}
