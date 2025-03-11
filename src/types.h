#ifndef DAHL_TYPES_H
#define DAHL_TYPES_H

#include <stdlib.h>

typedef double dahl_fp;

typedef struct
{
    size_t x;
    size_t y;
} shape2d;

typedef struct
{
    size_t x;
    size_t y;
    size_t z;
} shape3d;

#endif //!DAHL_TYPES_H
