#include "../include/dahl_types.h"
#include <math.h>
#include <stdio.h>

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
    printf("dahl_shape2d: x=%zu, y=%zu, z=%zu\n", shape.x, shape.y, shape.z);
}

void shape4d_print(dahl_shape4d const shape)
{
    printf("dahl_shape2d: x=%zu, y=%zu, z=%zu, t=%zu\n", shape.x, shape.y, shape.z, shape.t);
}

dahl_fp fp_round(dahl_fp const value, u_int8_t const precision)
{
    dahl_fp power = pow(10.0F, precision);
    return round(value * power) / power;
}
