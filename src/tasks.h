#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "types.h"
#include <starpu.h>

// Performs a x b = c, where:
// - x is the cross correlation operator
// - a, b and c are dahl_matrix objects
// - the shape of c must respect: c_nx = a_nx - b_nx + 1 and c_ny = a.y - b.y + 1
// - the shape of b should be smaller than the shape of a
void task_matrix_cross_correlation(dahl_matrix const* const a, dahl_matrix const* const b, dahl_matrix* const c);

// Performs max pooling on a and write output on b
void task_matrix_max_pooling(dahl_matrix const* const a, dahl_matrix* const b, size_t const pool_size);

// Apply relu function on each element of the block, i.e. max(elem i, 0)
void task_block_relu(dahl_block* const in);

// Sum the block values over the z axis and return it as a matrix of the same x,y shape.
dahl_matrix* task_block_sum_z_axis(dahl_block const* const in);

// Multiply each block value by the factor
void task_block_scal_self(dahl_block* const in, dahl_fp const factor);
void task_matrix_scal_self(dahl_matrix* const in, dahl_fp const factor);

// Performs a - b = c, where:
// - `-` is the value by value substraction
// - a, b and c are dahl_block objects of the same shape
// - c is created and returned by the function
dahl_block* task_block_sub(dahl_block const* const a, dahl_block const* const b);
dahl_matrix* task_matrix_sub(dahl_matrix const* const a, dahl_matrix const* const b);

// Performs a - b = a, where:
// - `-` is the value by value substraction
// - a and b are dahl_block objects of the same shape
// - a is modified by the function with the substraction result
void task_block_sub_self(dahl_block* const a, dahl_block const* const b);
void task_matrix_sub_self(dahl_matrix* const a, dahl_matrix const* const b);

// Performs a + b = c, where:
// - `+` is the value by value addition
// - a, b and c are dahl_block objects of the same shape
// - c is created and returned by the function
dahl_block* task_block_add(dahl_block const* const a, dahl_block const* const b);
dahl_block* task_matrix_add(dahl_matrix const* const a, dahl_matrix const* const b);

// Performs a + b = a, where:
// - `+` is the value by value addition
// - a and b are dahl_block objects of the same shape
// - a is modified by the function with the addition result
void task_block_add_self(dahl_block* const a, dahl_block const* const b);
void task_matrix_add_self(dahl_matrix* const a, dahl_matrix const* const b);


#endif //!DAHL_TASKS_H
