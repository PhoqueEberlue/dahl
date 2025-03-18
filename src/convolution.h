#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <starpu.h>
#include <stdlib.h>
#include <stddef.h>

#include "tasks.h"

typedef struct 
{
    const shape2d input_shape;

    const size_t num_filters;
    const size_t filter_size;

    // size * size * num_filters
    const shape3d filter_shape;
    const shape3d output_shape;

    dahl_block* filters;
    dahl_block* biases;
} convolution;

convolution create_convolution(shape2d input_shape, size_t filter_size, size_t num_filters);
dahl_block* forward_pass(convolution conv, dahl_matrix const* const input);
dahl_matrix* backward_pass(convolution conv, dahl_block* const dl_dout, double const learning_rate, dahl_matrix const* const input);

#endif //!DAHL_CONVOLUTION_H
