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

bool shape2d_equals(dahl_shape2d const a, dahl_shape2d const b);
bool shape3d_equals(dahl_shape3d const a, dahl_shape3d const b);
void shape2d_print(dahl_shape2d const shape);
void shape3d_print(dahl_shape3d const shape);

#endif //!DAHL_BASIC_TYPES_H
