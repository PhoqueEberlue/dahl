#ifndef DAHL_BASIC_TYPES_H
#define DAHL_BASIC_TYPES_H

#include "sys/types.h"
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

typedef struct _dahl_arena dahl_arena;

bool shape2d_equals(dahl_shape2d const a, dahl_shape2d const b);
bool shape3d_equals(dahl_shape3d const a, dahl_shape3d const b);
void shape2d_print(dahl_shape2d const shape);
void shape3d_print(dahl_shape3d const shape);

dahl_fp fp_round(dahl_fp const value, u_int8_t const precision);

dahl_arena* arena_new(size_t const size);
void* arena_put(dahl_arena* arena, size_t size);
void arena_reset(dahl_arena* arena);
void arena_delete(dahl_arena* arena);


#endif //!DAHL_BASIC_TYPES_H
