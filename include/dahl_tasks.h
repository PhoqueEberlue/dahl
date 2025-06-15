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
void task_block_sum_z_axis(dahl_block const* in, dahl_matrix* out);
void task_block_relu(dahl_block const* in, dahl_block* out);
void task_block_scal(dahl_block const* in, dahl_block* out, dahl_fp const factor);
void task_block_sub(dahl_block const* a, dahl_block const* b, dahl_block* c);
void task_block_add(dahl_block const* a, dahl_block const* b, dahl_block* c);
void task_block_add_value(dahl_block const* in, dahl_block* out, dahl_fp const value);
void task_block_sub_value(dahl_block const* in, dahl_block* out, dahl_fp const value);
void task_block_clip(dahl_block const* in, dahl_block* out, dahl_fp const min, dahl_fp const max);
dahl_fp task_block_sum(dahl_block const* in);
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

dahl_vector* task_matrix_sum_y_axis(dahl_matrix const* in);

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

void task_matrix_relu(dahl_matrix const* in, dahl_matrix* out);
void task_matrix_scal(dahl_matrix const* in, dahl_matrix* out, dahl_fp const factor);

void task_matrix_sub(dahl_matrix const* a, dahl_matrix const* b, dahl_matrix* c);
void task_matrix_add(dahl_matrix const* a, dahl_matrix const* b, dahl_matrix* c);

void task_matrix_add_value(dahl_matrix const* in, dahl_matrix* out, dahl_fp const value);
void task_matrix_sub_value(dahl_matrix const* in, dahl_matrix* out, dahl_fp const value);
void task_matrix_clip(dahl_matrix const* in, dahl_matrix* out, dahl_fp const min, dahl_fp const max);

dahl_fp task_matrix_sum(dahl_matrix const* in);

// Flatten a matrix and consider it as a row matrix
void task_matrix_to_flat_row(dahl_matrix* mat);

// Flatten a matrix and consider it as a column matrix
void task_matrix_to_flat_col(dahl_matrix* mat);
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

void task_vector_relu(dahl_vector const* in, dahl_vector* out);

void task_vector_scal(dahl_vector const* in, dahl_vector* out, dahl_fp const factor);

void task_vector_sub(dahl_vector const* a, dahl_vector const* b, dahl_vector* c);
void task_vector_add(dahl_vector const* a, dahl_vector const* b, dahl_vector* c);

void task_vector_add_value(dahl_vector const* in, dahl_vector* out, dahl_fp const value);
void task_vector_sub_value(dahl_vector const* in, dahl_vector* out, dahl_fp const value);
void task_vector_clip(dahl_vector const* in, dahl_vector* out, dahl_fp const min, dahl_fp const max);
dahl_fp task_vector_sum(dahl_vector const* in);
// ---------------------------- HELPER MACRO FOR TASKS COMMON TO ANY TYPES ----------------------------

// Type comparison without taking into account const qualifiers
#define TYPES_MATCH(T1, T2) \
    (__builtin_types_compatible_p(typeof(*(T1)), typeof(*(T2))))

#define TASK_RELU(IN, OUT)                                 \
    _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                   "IN and OUT must be of the same type"); \
    _Generic((OUT),                                        \
        dahl_block*: task_block_relu,                      \
        dahl_matrix*: task_matrix_relu,                    \
        dahl_vector*: task_vector_relu                     \
    )(IN, OUT)

#define TASK_RELU_SELF(SELF) TASK_RELU(SELF, SELF)

#define TASK_SCAL(IN, OUT, FACTOR)                         \
    _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                   "IN and OUT must be of the same type"); \
    _Generic((OUT),                                        \
        dahl_block*: task_block_scal,                      \
        dahl_matrix*: task_matrix_scal,                    \
        dahl_vector*: task_vector_scal                     \
    )(IN, OUT, FACTOR)

#define TASK_SCAL_SELF(SELF, FACTOR) TASK_SCAL(SELF, SELF, FACTOR)

// Performs `c` = `a` - `b`, where:
// - `-` is the value by value substraction
// - `a`, `b` and `c` are dahl_any objects of the same shape
#define TASK_SUB(A, B, C)                               \
    _Static_assert(TYPES_MATCH((A), (B)),               \
                   "A and B must be of the same type"); \
    _Static_assert(TYPES_MATCH((A), (C)),               \
                   "A and C must be of the same type"); \
    _Generic((C),                                       \
        dahl_block*: task_block_sub,                    \
        dahl_matrix*: task_matrix_sub,                  \
        dahl_vector*: task_vector_sub                   \
    )(A, B, C)

// Performs a_self -= b, where:
// - `-` is the value by value substraction
// - `a_self` and `b` are dahl data structures objects of the same type and shape
// - `a_self` is modified by the function with the substraction result
#define TASK_SUB_SELF(A_SELF, B) TASK_SUB(A_SELF, B, A_SELF)

// Performs `c` = `a` + `b`, where:
// - `+` is the value by value addition
// - `a`, `b` and `c` are dahl_any objects of the same shape
#define TASK_ADD(A, B, C)                               \
    _Static_assert(TYPES_MATCH((A), (B)),               \
                   "A and B must be of the same type"); \
    _Static_assert(TYPES_MATCH((A), (C)),               \
                   "A and C must be of the same type"); \
    _Generic((C),                                       \
        dahl_block*: task_block_add,                    \
        dahl_matrix*: task_matrix_add,                  \
        dahl_vector*: task_vector_add                   \
    )(A, B, C)

// Performs `a_self` += `b`, where:
// - `+` is the value by value addition
// - `a_self` and `b` are dahl_any objects of the same shape
// - `a_self` is modified by the function with the addition result
#define TASK_ADD_SELF(A_SELF, B) TASK_ADD(A_SELF, B, A_SELF)

// Add `value` to every elements of `in` and put the result in `out`
#define TASK_ADD_VALUE(IN, OUT, VALUE)                     \
    _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                   "IN and OUT must be of the same type"); \
    _Generic((OUT),                                        \
        dahl_block*: task_block_add_value,                 \
        dahl_matrix*: task_matrix_add_value,               \
        dahl_vector*: task_vector_add_value                \
    )(IN, OUT, VALUE)

// Add `value` to every elements of `self` writing directly in the same buffer
#define TASK_ADD_VALUE_SELF(SELF, VALUE) TASK_ADD_VALUE(SELF, SELF, VALUE)

// Substract `value` to every elements of `in` and put the result in `out`
#define TASK_SUB_VALUE(IN, OUT, VALUE)                     \
    _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                   "IN and OUT must be of the same type"); \
    _Generic((OUT),                                        \
        dahl_block*: task_block_sub_value,                 \
        dahl_matrix*: task_matrix_sub_value,               \
        dahl_vector*: task_vector_sub_value                \
    )(IN, OUT, VALUE)

// Substract `value` to every elements of `in` writing directly in the same buffer
#define TASK_SUB_VALUE_SELF(SELF, VALUE) TASK_SUB_VALUE(SELF, SELF, VALUE)

#define TASK_CLIP(IN, OUT, MIN, MAX)                       \
    _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                   "IN and OUT must be of the same type"); \
    _Generic((OUT),                                        \
        dahl_block*: task_block_clip,                      \
        dahl_matrix*: task_matrix_clip,                    \
        dahl_vector*: task_vector_clip                     \
    )(IN, OUT, MIN, MAX)

#define TASK_CLIP_SELF(SELF, MIN, MAX) TASK_CLIP(SELF, SELF, MIN, MAX)

#endif //!DAHL_TASKS_H
