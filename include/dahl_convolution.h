#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <stdlib.h>
#include <stddef.h>

#include "dahl_tasks.h"

typedef struct
{
    // img width * img height * batch size
    dahl_shape3d input_shape;

    size_t num_filters;
    size_t filter_size;

    // size * size * num_filters
    dahl_shape3d filter_shape;
    dahl_shape4d output_shape;

    dahl_block* filters;
    dahl_block* biases;

    // Arena for temporary data 
    // FIX If the convolution structure itself is allocated in an arena, it can't
    // free the scratch arena (or we must call a finalize method which break the principle of the arena).
    // So In fact here it's pretty ok to manage the layers structures with a regular malloc free pattern.
    // Or, we should provide at initialization as an argument, a scratch arena.
    // Other question, why should layers structures be heap allocated?
    dahl_arena* scratch_arena;
} dahl_convolution;

// Initialize a convolution structure.
dahl_convolution* convolution_init(dahl_arena*, dahl_shape3d input_shape, size_t filter_size, size_t num_filters);
dahl_tensor* convolution_forward(dahl_arena*, dahl_convolution*, dahl_block const* input_batch);
dahl_block* convolution_backward(dahl_arena*, dahl_convolution*, dahl_tensor const* dl_dout_batch, 
                                 double learning_rate, dahl_block const* input_batch);

#endif //!DAHL_CONVOLUTION_H
