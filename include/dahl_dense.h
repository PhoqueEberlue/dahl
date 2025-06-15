#ifndef DAHL_DENSE_H
#define DAHL_DENSE_H

#include "dahl_tasks.h"

typedef struct 
{
    dahl_shape3d const input_shape;
    size_t const output_size;

    dahl_block* input_data;
    dahl_vector* output;
    dahl_block* dl_dinput;

    dahl_block* weights;
    dahl_vector* biases;
} dahl_dense;

dahl_dense* dense_init(dahl_shape3d const input_shape, size_t const output_size);
dahl_vector* dense_forward(dahl_dense* dense, dahl_block* input_data);
dahl_block* dense_backward(dahl_dense* dense, dahl_vector const* dl_dout, dahl_fp const learning_rate);

#endif //!DAHL_DENSE_H
