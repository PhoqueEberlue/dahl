#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "types.h"
#include <starpu.h>

// Performs a x b = c, where:
// - x is the cross correlation operator
// - a, b and c are dahl_matrix objects
// - the shape of c must respect: c_nx = a_nx - b_nx + 1 and c_ny = a.y - b.y + 1
// - the shape of b should be smaller than the shape of a
void task_cross_correlation_2d(dahl_matrix const* const a, dahl_matrix const* const b, dahl_matrix* const c);

// Apply relu function on each element of the block, i.e. max(elem i, 0)
void task_relu(dahl_block* const in);

// Sum the block values over the z axis and return it as a matrix of the same x,y shape.
dahl_matrix* task_sum_z_axis(dahl_block const* const in);

// Multiply each block value by the factor
void task_scal(dahl_block* const in, dahl_fp const factor);

// Performs a - b = c, where:
// - `-` is the value by value substraction
// - a, b and c are dahl_block objects of the same shape
// - c is created and returned by the function
dahl_block* task_sub(dahl_block const* const a, dahl_block const* const b);

// Performs a + b = c, where:
// - `+` is the value by value addition
// - a, b and c are dahl_block objects of the same shape
// - c is created and returned by the function
dahl_block* task_add(dahl_block const* const a, dahl_block const* const b);

#endif //!DAHL_TASKS_H
