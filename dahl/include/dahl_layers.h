#ifndef DAHL_LAYERS_H
#define DAHL_LAYERS_H

#include <stdlib.h>
#include <stddef.h>
#include "dahl_tasks.h"

// ------------------------------------- Convolution -----------------------------------------------
typedef struct
{
    // img width * img height * nb channels * batch size
    dahl_shape4d input_shape;

    size_t num_filters;
    size_t filter_size;

    // size * size * num_filters
    dahl_shape4d filter_shape;
    dahl_shape4d output_shape;

    dahl_tensor* filters;
    dahl_vector* biases;

    dahl_arena* scratch_arena;
} dahl_convolution;

// Initialize a convolution structure.
dahl_convolution* convolution_init(dahl_arena* arena, dahl_arena* scratch_arena, 
                                   dahl_shape4d input_shape,
                                   size_t filter_size, size_t num_filters);

dahl_tensor* convolution_forward(dahl_arena*, dahl_convolution*, dahl_tensor const* input_batch);
dahl_tensor* convolution_backward(dahl_arena*, dahl_convolution*, dahl_tensor const* dl_dout_batch, 
                                  double learning_rate, dahl_tensor const* input_batch);

// ------------------------------------- Pooling ---------------------------------------------------
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

// ------------------------------------- Dense -----------------------------------------------------
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

dahl_dense* dense_init(dahl_arena* arena, dahl_arena* scratch_arena,
                       dahl_shape2d input_shape, size_t out_features);

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

// ------------------------------------- Relu ------------------------------------------------------
typedef struct 
{
    // Input shape with a batch dimension
    dahl_shape4d input_shape;

    // Mask that stores the index of positive values in the forward pass.
    // Useful for the backward pass so that we don't check again for negative values.
    dahl_tensor* mask_batch;
} dahl_relu;

dahl_relu* relu_init(dahl_arena* arena, dahl_shape4d input_shape);
void relu_forward(dahl_relu* relu, dahl_tensor* input_batch);
void relu_backward(dahl_relu* relu, dahl_tensor* dl_dout_batch);

#endif //!DAHL_LAYERS_H
