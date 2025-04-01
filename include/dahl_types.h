#ifndef DAHL_BASIC_TYPES_H
#define DAHL_BASIC_TYPES_H

#include <stddef.h>

typedef double dahl_fp;

typedef struct
{
    size_t x;
    size_t y;
    size_t z;
} dahl_shape3d;

typedef struct
{
    size_t x;
    size_t y;
} dahl_shape2d;

#endif //!DAHL_BASIC_TYPES_H
