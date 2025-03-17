#include "convolution.h"
#include "starpu_data.h"
#include "utils.h"
#include "tasks.h"

convolution create_convolution(shape2d input_shape, size_t filter_size, size_t num_filters)
{
    shape3d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = num_filters,
    };
    
    starpu_data_handle_t filters_handle = block_init(filter_shape);
    block_fill_random(filters_handle);

    shape3d output_shape = {
        .x = input_shape.x - filter_size + 1,
        .y = input_shape.y - filter_size + 1,
        .z = num_filters,
    };

    starpu_data_handle_t biases_handle = block_init(output_shape);
    block_fill_random(biases_handle);

    convolution conv = {
        .input_shape = input_shape,
        .num_filters = num_filters,
        .filter_size = filter_size,
        .filter_shape = filter_shape,
        .output_shape = output_shape,
        .filters_handle = filters_handle,
        .biases_handle = biases_handle,
    };

    return conv;
}

starpu_data_handle_t forward_pass(convolution conv, const starpu_data_handle_t input_handle)
{
    starpu_data_handle_t output_handle = block_init(conv.output_shape);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = conv.num_filters
	};

	starpu_data_partition(output_handle, &f);
	starpu_data_partition(conv.filters_handle, &f);

    for (int i = 0; i < conv.num_filters; i++)
    {
		starpu_data_handle_t sub_output = starpu_data_get_sub_data(output_handle, 1, i);
		const starpu_data_handle_t sub_filters = starpu_data_get_sub_data(conv.filters_handle, 1, i);

        task_cross_correlation_2d(input_handle, sub_filters, sub_output);
    }
    
	starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(conv.filters_handle, STARPU_MAIN_RAM);

    task_relu(output_handle);

    // Could be interesting to know if the relu task is really waiting for other tasks because starting?
    // It should be the case because of the data dependency and because it is working but we may verify that
    starpu_task_wait_for_all();

    return output_handle;
}

starpu_data_handle_t backward_pass(convolution conv, starpu_data_handle_t dl_dout, const double learning_rate, const starpu_data_handle_t input_handle)
{
    shape3d input_shape3d = { .x = conv.input_shape.x, .y = conv.input_shape.y, .z = conv.num_filters };
    // derivative loss
    starpu_data_handle_t dl_dinput = block_init(input_shape3d);
    starpu_data_handle_t dl_dfilters = block_init(conv.output_shape);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = conv.num_filters
	};

	starpu_data_partition(dl_dout, &f);
	starpu_data_partition(dl_dfilters, &f);
	starpu_data_partition(conv.filters_handle, &f);
	starpu_data_partition(dl_dinput, &f);

    for (int i = 0; i < conv.num_filters; i++)
    {
		const starpu_data_handle_t sub_dl_dout = starpu_data_get_sub_data(dl_dout, 1, i);
		starpu_data_handle_t sub_dl_dfilters = starpu_data_get_sub_data(dl_dfilters, 1, i);

        task_cross_correlation_2d(input_handle, sub_dl_dout, sub_dl_dfilters);

        // TODO: this doesn't work because here I need to increment sub_dl_dinput normally
        // dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")
        // do a partial res block and call add task?
        const starpu_data_handle_t sub_filters = starpu_data_get_sub_data(conv.filters_handle, 1, i);
        starpu_data_handle_t sub_dl_dinput = starpu_data_get_sub_data(dl_dinput, 1, i);
        
        task_cross_correlation_2d(sub_dl_dout, sub_filters, sub_dl_dinput);
    }

    starpu_task_wait_for_all();

    starpu_data_unpartition(dl_dout, STARPU_MAIN_RAM);
	starpu_data_unpartition(dl_dfilters, STARPU_MAIN_RAM);
	starpu_data_unpartition(conv.filters_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(dl_dinput, STARPU_MAIN_RAM); 
    
    // Updating filters and biases
    // filters -= dl_dfilters * learning_rate
    // biases -= dl_dout * learning_rate
    task_scal(dl_dfilters, learning_rate);
    task_scal(dl_dout, learning_rate);
    task_sub(conv.filters_handle, dl_dfilters);
    task_sub(conv.biases_handle, dl_dout);

    starpu_task_wait_for_all();

    // Then don't forget to compute the sum of dl_dfilters
    shape3d res_shape3d = { .x = conv.input_shape.x, .y = conv.input_shape.y, .z = 1 };
    starpu_data_handle_t reduced_dl_dinput = block_init(res_shape3d);

    task_sum_z_axis(dl_dinput, reduced_dl_dinput);

    return reduced_dl_dinput;
}
