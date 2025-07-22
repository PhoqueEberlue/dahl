#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <stdlib.h>
#include <stddef.h>

#include "dahl_tasks.h"

typedef struct
{
    // img width * img height * nb channels * batch size
    dahl_shape4d input_shape;

    size_t num_filters;
    size_t filter_size;

    // size * size * num_filters
    dahl_shape3d filter_shape;
    dahl_shape4d output_shape;

    dahl_block* filters;
    dahl_block* biases;

    dahl_arena* scratch_arena;
} dahl_convolution;

// Initialize a convolution structure.
dahl_convolution* convolution_init(dahl_arena* arena, dahl_arena* scratch_arena, 
                                   dahl_shape4d input_shape, size_t filter_size, size_t num_filters);
dahl_tensor* convolution_forward(dahl_arena*, dahl_convolution*, dahl_tensor const* input_batch);
dahl_tensor* convolution_backward(dahl_arena*, dahl_convolution*, dahl_tensor const* dl_dout_batch, 
                                 double learning_rate, dahl_tensor const* input_batch);

#endif //!DAHL_CONVOLUTION_H
