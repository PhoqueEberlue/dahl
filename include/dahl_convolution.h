#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <stdlib.h>
#include <stddef.h>

#include "dahl_tasks.h"

typedef struct
{
    // img width * img height * batch size
    dahl_shape3d input_shape;
    // Last batch of input data. Note: the pointer can be modified, the data cannot.
    dahl_block* input_batch;

    size_t num_filters;
    size_t filter_size;

    // size * size * num_filters
    dahl_shape3d filter_shape;
    dahl_shape4d output_shape;

    // Forward output data. Overwritten each convolution_forward() call
    dahl_tensor* output_batch;
    // Derivative of the backward input. Overwritten each convolution_backward() call
    dahl_block* dl_dinput_batch;

    dahl_block* filters;
    dahl_block* biases;
} dahl_convolution;

dahl_convolution* convolution_init(dahl_shape3d input_shape, size_t filter_size, size_t num_filters);
dahl_tensor* convolution_forward(dahl_convolution* conv, dahl_block const* input_batch);
dahl_block* convolution_backward(dahl_convolution* conv, dahl_tensor const* dl_dout_batch, double learning_rate);

#endif //!DAHL_CONVOLUTION_H
