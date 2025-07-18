#ifndef DAHL_POOLING_H
#define DAHL_POOLING_H

#include "dahl_data.h"

typedef struct 
{
    size_t pool_size;

    // Input shape with a batch dimension
    dahl_shape4d input_shape;

    // Mask that stores the max index values of each pooling window.
    // Useful to prevent recomputing the windows in the backward pass.
    dahl_tensor* mask_batch;

    dahl_shape4d output_shape;

} dahl_pooling;

dahl_pooling* pooling_init(dahl_arena*, size_t pool_size, dahl_shape4d input_shape);
dahl_tensor* pooling_forward(dahl_arena*, dahl_pooling*, dahl_tensor const* input_batch);
dahl_tensor* pooling_backward(dahl_arena*, dahl_pooling*, dahl_tensor const* dl_dout);

#endif //!DAHL_POOLING_H
