#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "types.h"
#include <starpu.h>

// ------------------------------------ TASKS FOR DAHL_BLOCK TYPE ------------------------------------
// Sum the block values over the z axis and return it as a matrix of the same x,y shape.
dahl_matrix* task_block_sum_z_axis(dahl_block const* const in);

// ------------------------------------ TASKS FOR DAHL_MATRIX TYPE ------------------------------------
// Performs `out` = `in` x `kernel`, where:
// - x is the cross correlation operator
// - `in`, `kernel` and `out` are dahl_matrix objects
// - the shape of `out` must respect: out_nx = in_nx - kernel_nx + 1 and out_ny = in_ny - kernel_ny + 1
// - the shape of the `kernel` should be smaller than the shape of `in` 
void task_matrix_cross_correlation(dahl_matrix const* const in, dahl_matrix const* const kernel, dahl_matrix* const out);

// Performs max pooling on `in`, write output on `out` and store mask of the max values indexes in `mask`
// - `out` shape should be equal to `in` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `in` shape.
void task_matrix_max_pooling(dahl_matrix const* const in, dahl_matrix* const out, dahl_matrix* const mask, size_t const pool_size);

// Performs a backward max pooling, copying each value of `in` into the right index of each window in `out` thanks to the `mask`.
// - `in` shape should be equal to `out` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `out` shape.
void task_matrix_backward_max_pooling(dahl_matrix const* const in, dahl_matrix const* const mask, dahl_matrix* const out, size_t const pool_size);

// Same as `task_matrix_backward_max_pooling` but stores the output directly in `mask_self`.
void task_matrix_backward_max_pooling_self(dahl_matrix const* const in, dahl_matrix* const mask_self, size_t const pool_size);

void task_matrix_softmax(dahl_matrix* const in);

// ------------------------------------ TASKS FOR DAHL_ANY TYPE ------------------------------------
// Apply relu function on each element of the `dahl_any`, i.e. max(elem i, 0)
void task_any_relu(dahl_any const in, dahl_any out);
void task_any_relu_self(dahl_any self);

// Multiply each block value by the factor
void task_any_scal(dahl_any const in, dahl_any const out,  dahl_fp const factor);
void task_any_scal_self(dahl_any self, dahl_fp const factor);

// Performs `c` = `a` - `b`, where:
// - `-` is the value by value substraction
// - `a`, `b` and `c` are dahl_any objects of the same shape
// - `c` is created and returned by the function
dahl_any task_any_sub(dahl_any const a, dahl_any const b);

// Performs a_self -= b, where:
// - `-` is the value by value substraction
// - `a_self` and `b` are dahl_any objects of the same shape
// - `a_self` is modified by the function with the substraction result
void task_any_sub_self(dahl_any a_self, dahl_any const b);

// Performs `c` = `a` + `b`, where:
// - `+` is the value by value addition
// - `a`, `b` and `c` are dahl_any objects of the same shape
// - `c` is created and returned by the function
dahl_any task_any_add(dahl_any const a, dahl_any const b);

// Performs `a_self` += `b`, where:
// - `+` is the value by value addition
// - `a_self` and `b` are dahl_any objects of the same shape
// - `a_self` is modified by the function with the addition result
void task_any_add_self(dahl_any a_self, dahl_any const b);

#endif //!DAHL_TASKS_H
