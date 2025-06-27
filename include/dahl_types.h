#ifndef DAHL_BASIC_TYPES_H
#define DAHL_BASIC_TYPES_H

#include "sys/types.h"
#include <stddef.h>

typedef double dahl_fp;

typedef struct
{
    size_t x;
    size_t y;
} dahl_shape2d;

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
    size_t z;
    size_t t;
} dahl_shape4d;

bool shape2d_equals(dahl_shape2d a, dahl_shape2d b);
bool shape3d_equals(dahl_shape3d a, dahl_shape3d b);
void shape2d_print(dahl_shape2d shape);
void shape3d_print(dahl_shape3d shape);
void shape4d_print(dahl_shape4d shape);

dahl_fp fp_round(dahl_fp value, u_int8_t precision);

#endif //!DAHL_BASIC_TYPES_H
