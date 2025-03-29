#ifndef DAHL_DENSE_H
#define DAHL_DENSE_H

#include "../types.h"

typedef struct 
{
    shape3d const input_shape;
    shape3d const output_shape;

    dahl_block* filters;
    dahl_block* biases;
} convolution;


#endif //!DAHL_DENSE_H
