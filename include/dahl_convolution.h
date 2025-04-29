#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <stdlib.h>
#include <stddef.h>

#include "dahl_tasks.h"

typedef struct
{
    dahl_shape2d const input_shape;
    // Last input data
    dahl_matrix* input_data;

    size_t const num_filters;
    size_t const filter_size;

    // size * size * num_filters
    dahl_shape3d const filter_shape;
    dahl_shape3d const output_shape;

    dahl_block* filters;
    dahl_block* biases;
} dahl_convolution;

dahl_convolution* convolution_init(dahl_shape2d input_shape, size_t filter_size, size_t num_filters);
dahl_block* convolution_forward(dahl_convolution* conv, dahl_matrix const* input);
dahl_matrix* convolution_backward(dahl_convolution* conv, dahl_block* dl_dout, double const learning_rate, dahl_matrix const* input);

#endif //!DAHL_CONVOLUTION_H
