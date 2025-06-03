#include "../include/dahl_types.h"
#include <stdio.h>

bool shape2d_equals(dahl_shape2d const a, dahl_shape2d const b)
{
    return (bool)(a.x == b.x && a.y == b.y);
}

bool shape3d_equals(dahl_shape3d const a, dahl_shape3d const b)
{
    return (bool)(a.x == b.x && a.y == b.y && a.z == b.z);
}

void shape2d_print(dahl_shape2d const shape)
{
    printf("dahl_shape2d: x=%zu, y=%zu\n", shape.x, shape.y);
}

void shape3d_print(dahl_shape3d const shape)
{
    printf("dahl_shape2d: x=%zu, y=%zu, z=%zu\n", shape.x, shape.y, shape.z);
}
