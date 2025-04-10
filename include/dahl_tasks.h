#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "dahl_data.h"

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

// Performs max dahl_pooling on `in`, write output on `out` and store mask of the max values indexes in `mask`
// - `out` shape should be equal to `in` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `in` shape.
void task_matrix_max_pooling(dahl_matrix const* const in, dahl_matrix* const out, dahl_matrix* const mask, size_t const pool_size);

// Performs a backward max dahl_pooling, copying each value of `in` into the right index of each window in `out` thanks to the `mask`.
// - `in` shape should be equal to `out` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `out` shape.
void task_matrix_backward_max_pooling(dahl_matrix const* const in, dahl_matrix const* const mask, dahl_matrix* const out, size_t const pool_size);

// Same as `task_matrix_backward_max_pooling` but stores the output directly in `mask_self`.
void task_matrix_backward_max_pooling_self(dahl_matrix const* const in, dahl_matrix* const mask_self, size_t const pool_size);

dahl_vector* task_matrix_vector_product(dahl_matrix const* const mat, dahl_vector const* const vec);

// ------------------------------------ TASKS FOR DAHL_VECTOR TYPE ------------------------------------
// Performs the softmax function with `in` vector and writes the result to `out`.
void task_vector_softmax(dahl_vector const* const in, dahl_vector* const out);

// Performs `a`  `b`, where:
// - `` is the dot product
// - `a`, `b` are dahl_vector of the same length
// - returns the result as a dahl_fp
dahl_fp task_vector_dot_product(dahl_vector const* const a, dahl_vector const* const b);

// Create and return a diagonal dahl_matrix of the input dahl_vector
dahl_matrix* task_vector_diag(dahl_vector const* const in);

dahl_matrix* task_vector_softmax_derivative(dahl_vector const* const in);

// ------------------------------------ TASKS FOR DAHL_ANY TYPE ------------------------------------
// Helper to wrap a dahl data structure into a `dahl_any`.
// Initialize a stack allocated `dahl_any` object from a `dahl_block*`, `dahl_matrix*` or `dahl_vector*`.
#define AS_ANY(X) _Generic((X),                               \
        dahl_block*:                                          \
            (dahl_any)                                        \
            {                                                 \
                .structure = { .block = (dahl_block*)(X) },   \
                .type = dahl_type_block                       \
            },                                                \
        dahl_matrix*:                                         \
            (dahl_any)                                        \
            {                                                 \
                .structure = { .matrix = (dahl_matrix*)(X) }, \
                .type = dahl_type_matrix                      \
            },                                                \
        dahl_vector*:                                         \
            (dahl_any)                                        \
            {                                                 \
                .structure = { .vector = (dahl_vector*)(X) }, \
                .type = dahl_type_vector                      \
            }                                                 \
    )   // TODO: is `default` required?

// Helper to unwrap a `dahl_any`, to be used for functions that take and return the same 
// dahl data structure types using `dahl_any` wrapper.
// Gets `dahl_block*`, `dahl_matrix*` or `dahl_vector*` from `OUT` by reading `IN`'s type
#define FROM_ANY(IN, OUT) _Generic((IN), \
        dahl_block*:                     \
            (dahl_block*)                \
            {                            \
                (OUT).structure.block    \
            },                           \
        dahl_matrix*:                    \
            (dahl_matrix*)               \
            {                            \
                (OUT).structure.matrix   \
            },                           \
        dahl_vector*:                    \
            (dahl_vector*)               \
            {                            \
                (OUT).structure.vector   \
            }                            \
    )   // TODO: is `default` required?

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

#endif //!DAHL_TASKS_H
