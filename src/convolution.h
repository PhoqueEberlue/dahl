#ifndef DAHL_CONVOLUTION_H
#define DAHL_CONVOLUTION_H

#include <starpu.h>
#include <stdlib.h>
#include <stddef.h>

#include "types.h"

typedef struct 
{
    const shape2d input_shape;

    const size_t num_filters;
    const size_t filter_size;
    // size * size * num_filters
    const shape3d filter_shape;
    const shape3d output_shape;

    starpu_data_handle_t filters_handle;
    starpu_data_handle_t biases_handle;
} convolution;

convolution create_convolution(shape2d input_shape, size_t filter_size, size_t num_filters);
starpu_data_handle_t forward_pass(convolution conv, starpu_data_handle_t input_handle);


#endif //!DAHL_CONVOLUTION_H
