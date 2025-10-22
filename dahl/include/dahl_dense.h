#ifndef DAHL_DENSE_H
#define DAHL_DENSE_H

#include "dahl_tasks.h"

typedef struct 
{
    dahl_shape2d input_shape;
    dahl_shape2d output_shape;
    dahl_shape2d weights_shape;

    dahl_matrix* weights;
    dahl_vector* biases;

    // Arena for temporary data
    dahl_arena* scratch_arena;
} dahl_dense;

dahl_dense* dense_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_shape2d input_shape, size_t out_features);

// Returns the prediction for each batch
dahl_matrix* dense_forward(dahl_arena*, dahl_dense*, dahl_matrix const* input_batch);

// `dl_dout` gradient batch of the last forward pass
//
// Parameters:
// - `dl_dout_batch` derivative output of the previous layer
// - `input_batch` the input from the last forward pass
// - `learning_rate`
dahl_matrix* dense_backward(dahl_arena*, dahl_dense*, dahl_matrix const* dl_dout_batch, 
                            dahl_matrix const* input_batch, dahl_fp learning_rate);

#endif //!DAHL_DENSE_H
