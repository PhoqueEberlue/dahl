#ifndef DAHL_DENSE_H
#define DAHL_DENSE_H

#include "dahl_tasks.h"

typedef struct 
{
    dahl_shape4d input_shape;
    dahl_shape2d output_shape;

    dahl_tensor* input_batch;
    dahl_matrix* output_batch;
    dahl_tensor* dl_dinput_batch;

    dahl_block* weights;
    dahl_vector* biases;
} dahl_dense;

dahl_dense* dense_init(dahl_shape4d input_shape, size_t n_classes);

// Returns the prediction for each batch
dahl_matrix* dense_forward(dahl_dense* dense, dahl_tensor const* input_batch);

// `dl_dout` gradient batch of the last forward pass
dahl_tensor* dense_backward(dahl_dense* dense, dahl_matrix const* dl_dout_batch, dahl_fp learning_rate);

#endif //!DAHL_DENSE_H
