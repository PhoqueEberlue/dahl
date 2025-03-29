#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <starpu.h>
#include <stdlib.h>
#include <stddef.h>

#include "../tasks.h"
#include "../types.h"

typedef struct
{
    shape2d const input_shape;
    // Last input data
    dahl_matrix* input_data;

    size_t const num_filters;
    size_t const filter_size;

    // size * size * num_filters
    shape3d const filter_shape;
    shape3d const output_shape;

    dahl_block* filters;
    dahl_block* biases;
} convolution;

convolution* convolution_init(shape2d input_shape, size_t filter_size, size_t num_filters);
dahl_block* convolution_forward(convolution* const conv, dahl_matrix const* const input);
dahl_matrix* convolution_backward(convolution* const conv, dahl_block* const dl_dout, double const learning_rate, dahl_matrix const* const input);

#endif //!DAHL_CONVOLUTION_H
