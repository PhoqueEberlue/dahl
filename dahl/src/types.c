#include "../include/dahl_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

bool shape2d_equals(dahl_shape2d const a, dahl_shape2d const b)
{
    return (bool)(a.x == b.x && a.y == b.y);
}

bool shape3d_equals(dahl_shape3d const a, dahl_shape3d const b)
{
    return (bool)(a.x == b.x && a.y == b.y && a.z == b.z);
}

bool shape4d_equals(dahl_shape4d const a, dahl_shape4d const b)
{
    return (bool)(a.x == b.x && a.y == b.y && a.z == b.z && a.t == b.t);
}

void shape2d_print(dahl_shape2d const shape)
{
    printf("dahl_shape2d: x=%zu, y=%zu\n", shape.x, shape.y);
}

void shape3d_print(dahl_shape3d const shape)
{
    printf("dahl_shape3d: x=%zu, y=%zu, z=%zu\n", shape.x, shape.y, shape.z);
}

void shape4d_print(dahl_shape4d const shape)
{
    printf("dahl_shape4d: x=%zu, y=%zu, z=%zu, t=%zu\n", shape.x, shape.y, shape.z, shape.t);
}

dahl_fp fp_round(dahl_fp const value, int8_t const precision)
{
    assert(precision > 0);
    dahl_fp power = pow(10.0F, precision);
    return round(value * power) / power;
}

bool fp_equals(dahl_fp a, dahl_fp b)
{
    return (bool)(a == b);
}

bool fp_equals_round(dahl_fp a, dahl_fp b, int8_t precision)
{
    return (bool)(fp_round(a, precision) == fp_round(b, precision));
}

dahl_fp fp_rand(dahl_fp min, dahl_fp max)
{
    // Generate a random number between 0 and 1
    dahl_fp rand_norm = (dahl_fp)rand() / (dahl_fp)(RAND_MAX);
    // Apply the range
    dahl_fp rand_val = ((rand_norm * (max - min)) + min);
    return rand_val;
}
