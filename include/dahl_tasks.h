#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "dahl_data.h"

// Naming convention <task>_<type>_<name>_<mode>:
// - `task` always mean that this function will be scheduled on GPU/CPU
// - <type>: `block`, `matrix` or `vector` indicate the primary data structure used in the task
// - <name>: name of the function
// - <mode> defines how the result is returned:
//   - `` default implementation, the user should instanciate the output data structure before calling the function
//   - `self` writes the result in the same data structure that is used for input (usually the argument named *_self)
//   - `init` the function instanciates and returns the result data structure

// ------------------------------------ TASKS FOR DAHL_BLOCK TYPE ------------------------------------
// Sum the block values over the z axis and return it as a matrix of the same x,y shape.
dahl_matrix* task_block_sum_z_axis(dahl_block const* in);

// ------------------------------------ TASKS FOR DAHL_MATRIX TYPE ------------------------------------
// Performs `out` = `in` x `kernel`, where:
// - x is the cross correlation operator
// - `in`, `kernel` and `out` are dahl_matrix objects
// - the shape of `out` must respect: out_nx = in_nx - kernel_nx + 1 and out_ny = in_ny - kernel_ny + 1
// - the shape of the `kernel` should be smaller than the shape of `in` 
void task_matrix_cross_correlation(dahl_matrix const* in, dahl_matrix const* kernel, dahl_matrix* out);

// Performs max dahl_pooling on `in`, write output on `out` and store mask of the max values indexes in `mask`
// - `out` shape should be equal to `in` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `in` shape.
void task_matrix_max_pooling(dahl_matrix const* in, dahl_matrix* out, dahl_matrix* mask, size_t const pool_size);

// Performs a backward max dahl_pooling, copying each value of `in` into the right index of each window in `out` thanks to the `mask`.
// - `in` shape should be equal to `out` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `out` shape.
void task_matrix_backward_max_pooling(dahl_matrix const* in, dahl_matrix const* mask, dahl_matrix* out, size_t const pool_size);

// Same as `task_matrix_backward_max_pooling` but stores the output directly in `mask_self`.
void task_matrix_backward_max_pooling_self(dahl_matrix const* in, dahl_matrix* mask_self, size_t const pool_size);

// Performs matrix vector product. Tries to find the right dimension to perform the operation.
void task_matrix_vector_product(dahl_matrix const* mat, dahl_vector const* vec, dahl_vector* out);
dahl_vector* task_matrix_vector_product_init(dahl_matrix const* mat, dahl_vector const* vec);

void task_matrix_matrix_product(dahl_matrix const* a, dahl_matrix const* b, dahl_matrix* c);
dahl_matrix* task_matrix_matrix_product_init(dahl_matrix const* a, dahl_matrix const* b);

void task_matrix_transpose(dahl_matrix const* in, dahl_matrix* out);
dahl_matrix* task_matrix_transpose_init(dahl_matrix const* in);

// ------------------------------------ TASKS FOR DAHL_VECTOR TYPE ------------------------------------
// Performs the softmax function with `in` vector and writes the result to `out`.
void task_vector_softmax(dahl_vector const* in, dahl_vector* out);

// Performs the softmax function with `in` vector and returns the result.
dahl_vector* task_vector_softmax_init(dahl_vector const* in);

// Performs `a`  `b`, where:
// - `` is the dot product
// - `a`, `b` are dahl_vector of the same length
// - returns the result as a dahl_fp
dahl_fp task_vector_dot_product(dahl_vector const* a, dahl_vector const* b);

// Create and return a diagonal dahl_matrix of the input dahl_vector
dahl_matrix* task_vector_diag(dahl_vector const* in);

dahl_matrix* task_vector_softmax_derivative(dahl_vector const* in);

dahl_fp task_vector_cross_entropy_loss(dahl_vector const* predictions, dahl_vector const* targets);

dahl_vector* task_vector_cross_entropy_loss_gradient(dahl_vector const* predictions, dahl_vector const* targets);

// ------------------------------------ TASKS FOR DAHL_ANY TYPE ------------------------------------
// Apply relu function on each element of the `dahl_any`, i.e. max(elem i, 0)
void task_relu(dahl_any const in, dahl_any out);
#define TASK_RELU(IN, OUT) task_relu(AS_ANY(IN), AS_ANY(OUT))
#define TASK_RELU_SELF(SELF) task_relu(AS_ANY(SELF), AS_ANY(SELF))

// Multiply each value by the factor
void task_scal(dahl_any const in, dahl_any out, dahl_fp const factor);
#define TASK_SCAL(IN, OUT, FACTOR) task_scal(AS_ANY(IN), AS_ANY(OUT), FACTOR)
#define TASK_SCAL_SELF(SELF, FACTOR) task_scal(AS_ANY(SELF), AS_ANY(SELF), FACTOR)

// Performs `c` = `a` - `b`, where:
// - `-` is the value by value substraction
// - `a`, `b` and `c` are dahl_any objects of the same shape
void task_sub(dahl_any const a, dahl_any const b, dahl_any c);
#define TASK_SUB(A, B, C) task_sub(AS_ANY(A), AS_ANY(B), AS_ANY(C))

// Performs a_self -= b, where:
// - `-` is the value by value substraction
// - `a_self` and `b` are dahl_any objects of the same shape
// - `a_self` is modified by the function with the substraction result
#define TASK_SUB_SELF(A_SELF, B) task_sub(AS_ANY(A_SELF), AS_ANY(B), AS_ANY(A_SELF))

// Performs `c` = `a` + `b`, where:
// - `+` is the value by value addition
// - `a`, `b` and `c` are dahl_any objects of the same shape
void task_add(dahl_any const a, dahl_any const b, dahl_any c);
#define TASK_ADD(A, B, C) task_add(AS_ANY(A), AS_ANY(B), AS_ANY(C))

// Performs `a_self` += `b`, where:
// - `+` is the value by value addition
// - `a_self` and `b` are dahl_any objects of the same shape
// - `a_self` is modified by the function with the addition result
#define TASK_ADD_SELF(A_SELF, B) task_add(AS_ANY(A_SELF), AS_ANY(B), AS_ANY(A_SELF))

// Add `value` to every elements of `in` and put the result in `out`
void task_add_value(dahl_any const in, dahl_any out, dahl_fp const value);
#define TASK_ADD_VALUE(IN, OUT, VALUE) task_add_value(AS_ANY(IN), AS_ANY(OUT), VALUE)

// Add `value` to every elements of `in` writing directly in the same buffer
#define TASK_ADD_VALUE_SELF(SELF, VALUE) task_add_value(AS_ANY(SELF), AS_ANY(SELF), VALUE)

// Substract `value` to every elements of `in` and put the result in `out`
void task_sub_value(dahl_any const in, dahl_any out, dahl_fp const value);
#define TASK_SUB_VALUE(IN, OUT, VALUE) task_sub_value(AS_ANY(IN), AS_ANY(OUT), VALUE)

// Substract `value` to every elements of `in` writing directly in the same buffer
#define TASK_SUB_VALUE_SELF(SELF, VALUE) task_sub_value(AS_ANY(SELF), AS_ANY(SELF), VALUE)

void task_clip(dahl_any const in, dahl_any const out, dahl_fp const min, dahl_fp const max);
#define TASK_CLIP(IN, OUT, MIN, MAX) task_clip(AS_ANY(IN), AS_ANY(OUT), MIN, MAX)

#define TASK_CLIP_SELF(SELF, MIN, MAX) task_clip(AS_ANY(SELF), AS_ANY(SELF), MIN, MAX)

#endif //!DAHL_TASKS_H
