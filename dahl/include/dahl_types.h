#ifndef DAHL_BASIC_TYPES_H
#define DAHL_BASIC_TYPES_H

#include "sys/types.h"
#include <stddef.h>

// Different types
typedef enum {
    DAHL_TENSOR,
    DAHL_BLOCK,
    DAHL_MATRIX,
    DAHL_VECTOR,
    DAHL_SCALAR,
} dahl_type;

typedef enum {
    DAHL_READ,
    DAHL_MUT,
    DAHL_REDUX,
} dahl_access;

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

bool fp_equals(dahl_fp a, dahl_fp b);
bool fp_equals_round(dahl_fp a, dahl_fp b, int8_t precision);
bool shape2d_equals(dahl_shape2d a, dahl_shape2d b);
bool shape3d_equals(dahl_shape3d a, dahl_shape3d b);
bool shape4d_equals(dahl_shape4d a, dahl_shape4d b);
void shape2d_print(dahl_shape2d shape);
void shape3d_print(dahl_shape3d shape);
void shape4d_print(dahl_shape4d shape);

dahl_fp fp_round(dahl_fp value, int8_t precision);
dahl_fp fp_rand(dahl_fp min, dahl_fp max);

#endif //!DAHL_BASIC_TYPES_H
