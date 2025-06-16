#include "../../include/dahl_dense.h"
#include "starpu_task.h"
#include <stdlib.h>
#include "../arena/arena.h"

dahl_dense* dense_init(dahl_shape3d const input_shape, size_t const output_size)
{
    // All the allocations in this function will be performed in the persistent arena
    dahl_arena* save_arena = context_arena;
    context_arena = default_arena;

    dahl_dense* dense = dahl_arena_alloc(sizeof(dahl_dense));

    dahl_vector* output = vector_init(output_size);

    dahl_block* dl_dinput = block_init(input_shape);

    *(dahl_shape3d*)&dense->input_shape = input_shape;
    *(size_t*)&dense->output_size = output_size;

    dahl_shape3d weights_shape = { .x = input_shape.x * input_shape.y, .y = output_size, .z = input_shape.z };

    dense->weights =  block_init_random(weights_shape);
    dense->biases =  vector_init_random(output_size);
    dense->output = output;
    dense->dl_dinput = dl_dinput;

    context_arena = save_arena;

    return dense;
}

dahl_vector* dense_forward(dahl_dense* dense, dahl_block* input_data)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* save_arena = context_arena;
    context_arena = temporary_arena;

    dense->input_data = input_data;

    // x: number of classes, y: number of channels
    dahl_shape2d tmp_shape = { .x = dense->output_size, .y = dense->input_shape.z };

    dahl_matrix* tmp = matrix_init(tmp_shape);

    block_partition_along_z_flat(input_data);
    block_partition_along_z(dense->weights);
    matrix_partition_along_y(tmp);

    size_t const n_channels = block_get_sub_matrix_nb(input_data);

    for (size_t i = 0; i < n_channels; i++)
    {
        dahl_vector const* sub_input_flatten = block_get_sub_vector(input_data, i);
        dahl_matrix const* sub_weights = block_get_sub_matrix(dense->weights, i);
        dahl_vector* sub_tmp = matrix_get_sub_vector(tmp, i);

        task_matrix_vector_product(sub_weights, sub_input_flatten, sub_tmp);
    }

    block_unpartition(input_data);
    block_unpartition(dense->weights);
    matrix_unpartition(tmp);

    dahl_vector* out = task_matrix_sum_y_axis(tmp);

    TASK_ADD_SELF(out, dense->biases);

    task_vector_softmax(out, dense->output);

    dahl_arena_reset(temporary_arena);
    context_arena = save_arena;

    return dense->output;
}

dahl_block* dense_backward(dahl_dense* dense, dahl_vector const* dl_dout, dahl_fp const learning_rate)
{
    // All the allocations in this function will be performed in the temporary arena
    dahl_arena* save_arena = context_arena;
    context_arena = temporary_arena;

    dahl_matrix const* tmp = task_vector_softmax_derivative(dense->output);

    dahl_vector* dl_dy = task_matrix_vector_product_init(tmp, dl_dout);

    // We need to clone dl_dy because we change its dimensions
    dahl_vector* dl_dy_clone = vector_clone(dl_dy);
    dahl_matrix const* dl_dy_col = vector_to_column_matrix(dl_dy_clone);

    dahl_shape3d dl_dw_shape = { .x = dense->input_shape.x * dense->input_shape.y, .y = dense->output_size, .z = dense->input_shape.z };
    dahl_block* dl_dw = block_init(dl_dw_shape);

    // Reset dl_dinput
    task_block_fill(dense->dl_dinput, 0);

    block_partition_along_z(dense->input_data);
    block_partition_along_z(dense->weights);
    block_partition_along_z(dl_dw);

    // Here we partition the output into flat vectors, because the output is stored as multiple matrices,
    // however we need to perform the partial computation on flattened views of the matrices.
    block_partition_along_z_flat(dense->dl_dinput);

    size_t const n_channels = block_get_sub_matrix_nb(dense->input_data);

    for (size_t i = 0; i < n_channels; i++)
    {
        dahl_matrix* sub_input = block_get_sub_matrix(dense->input_data, i);
        task_matrix_to_flat_row(sub_input);

        dahl_matrix* sub_dl_dw = block_get_sub_matrix(dl_dw, i);

        task_matrix_matrix_product(dl_dy_col, sub_input, sub_dl_dw);

        dahl_matrix* sub_weights = block_get_sub_matrix(dense->weights, i);
        dahl_matrix const* sub_weights_t = task_matrix_transpose_init(sub_weights);

        dahl_vector* sub_output = block_get_sub_vector(dense->dl_dinput, i);
        task_matrix_vector_product(sub_weights_t, dl_dy, sub_output);

        // Updating weights
        TASK_SCAL_SELF(sub_dl_dw, learning_rate);
        TASK_SUB_SELF(sub_weights, sub_dl_dw);
    }

    block_unpartition(dense->input_data);
    block_unpartition(dense->weights);
    block_unpartition(dl_dw);
    block_unpartition(dense->dl_dinput);

    // Updating biases
    TASK_SCAL_SELF(dl_dy, learning_rate);
    TASK_SUB_SELF(dense->biases, dl_dy);

    dahl_arena_reset(temporary_arena);
    context_arena = save_arena;

    return dense->dl_dinput;
}
