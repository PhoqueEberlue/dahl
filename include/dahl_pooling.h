#ifndef DAHL_POOLING_H
#define DAHL_POOLING_H

#include "dahl_data.h"

typedef struct 
{
    size_t const pool_size;

    // Input shape with a batch dimension
    dahl_shape4d const input_shape;

    // Mask storing the max values indexes from the last input data, act as dl_dinput
    dahl_tensor* mask_batch;

    dahl_shape4d const output_shape;

    // Forward output data. Overwritten each pooling_forward() call
    dahl_tensor* output_batch;

} dahl_pooling;

dahl_pooling* pooling_init(size_t pool_size, dahl_shape4d input_shape);
dahl_tensor* pooling_forward(dahl_pooling* pool, dahl_tensor* input_batch);
dahl_tensor* pooling_backward(dahl_pooling* pool, dahl_tensor* dl_dout);

#endif //!DAHL_POOLING_H
