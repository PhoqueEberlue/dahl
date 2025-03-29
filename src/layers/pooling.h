#ifndef DAHL_POOLING_H
#define DAHL_POOLING_H

#include "../types.h"
#include "../tasks.h"

typedef struct 
{
    size_t const pool_size;

    // not to be confound with convolution's input data which is 2d image,
    // here it's the result of the previous layer, so a 3d matrix (conv.forward return value)
    shape3d input_shape;
    // Last input data
    dahl_block* input_data;
    // Mask storing the max values indexes from the last input data
    dahl_block* mask;

    shape3d output_shape;
    dahl_block* output_data;
} pooling;

pooling* pooling_init(size_t const pool_size);
dahl_block* pooling_forward(pooling* const pool, dahl_block const* const input);
dahl_block* pooling_backward(pooling* const pool, dahl_block const* const dl_dout);

#endif //!DAHL_POOLING_H
