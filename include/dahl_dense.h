#ifndef DAHL_DENSE_H
#define DAHL_DENSE_H

#include "dahl_data.h"

typedef struct 
{
    dahl_shape3d const input_shape;
    dahl_shape3d const output_shape;

    dahl_block* filters;
    dahl_block* biases;
} dahl_dense;


#endif //!DAHL_DENSE_H
