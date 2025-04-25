#ifndef DAHL_POOLING_H
#define DAHL_POOLING_H

#include "dahl_data.h"

typedef struct 
{
    size_t const pool_size;

    // not to be confound with dahl_convolution's input data which is 2d image,
    // here it's the result of the previous layer, so a 3d matrix (conv.forward return value)
    dahl_shape3d const input_shape;
    // Last input data
    dahl_block* input_data;
    // Mask storing the max values indexes from the last input data
    dahl_block* mask;

    dahl_shape3d const output_shape;
    dahl_block* output_data;
} dahl_pooling;

dahl_pooling* pooling_init(size_t const pool_size, dahl_shape3d const input_shape);
dahl_block* pooling_forward(dahl_pooling* const pool, dahl_block const* input);
dahl_block* pooling_backward(dahl_pooling* const pool, dahl_block* const dl_dout);

#endif //!DAHL_POOLING_H
