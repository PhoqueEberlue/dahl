#include "../../include/dahl_dense.h"
#include <stdlib.h>

dahl_dense* dense_init(dahl_shape3d const input_shape, size_t const output_size)
{
    dahl_dense* dense = malloc(sizeof(dahl_dense));

    *(dahl_shape3d*)&dense->input_shape = input_shape;
    *(size_t*)&dense->output_size = output_size;

    dahl_shape2d weights_shape = { .x = input_shape.x * input_shape.y * input_shape.z, .y = output_size };

    dense->weights =  matrix_init_random(weights_shape);
    dense->biases =  vector_init_random(output_size);

    return dense;
}

dahl_vector* dense_forward(dahl_dense* dense, dahl_block* input_data)
{
    // TODO: here it deletes the original input_data pointer which makes it unavailable for the caller, not good right?
    dense->input_data_flattened = block_to_vector(input_data);
    dahl_vector* tmp = task_matrix_vector_product_init(dense->weights, dense->input_data_flattened);
    TASK_ADD_SELF(tmp, dense->biases);

    dense->output = task_vector_softmax_init(tmp);

    vector_finalize(tmp);

    return dense->output;
}

dahl_block* dense_backward(dahl_dense* dense, dahl_vector const* dl_dout, dahl_fp const learning_rate)
{
    dahl_matrix const* tmp = task_vector_softmax_derivative(dense->output);

    dahl_vector const* dl_dy = task_matrix_vector_product_init(tmp, dl_dout);

    dahl_vector* dl_dy_clone = vector_clone(dl_dy);

    // We need to clone dl_dy because its dissapearing here
    dahl_matrix* dl_dy_mat = vector_to_column_matrix(dl_dy_clone);
    // FIX Here it deletes the data pointed by dense->input_data_flattened, which is *fine* for this code but not good in general
    dahl_matrix* input_mat = vector_to_row_matrix(dense->input_data_flattened);

    dahl_matrix* dl_dw = task_matrix_matrix_product_init(dl_dy_mat, input_mat);

    dahl_matrix* weights_t = task_matrix_transpose_init(dense->weights);

    dahl_vector* dl_dinput = task_matrix_vector_product_init(weights_t, dl_dy);

    dahl_block* dl_dinput_block = vector_to_block(dl_dinput, dense->input_shape);

    TASK_SCAL_SELF(dl_dw, learning_rate);

    dahl_vector* tmp_2 = vector_init(vector_get_len(dl_dy));
    TASK_SCAL(dl_dy, tmp_2, learning_rate);

    TASK_SUB_SELF(dense->weights, dl_dw);
    TASK_SUB_SELF(dense->biases, tmp_2);

    return dl_dinput_block;
}
