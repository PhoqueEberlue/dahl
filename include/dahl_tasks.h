#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "dahl_data.h"

// Naming convention <task>_<type>_<name>_<mode>:
// - `task` always mean that this function will be scheduled on GPU/CPU
// - <type>: `block`, `matrix` or `vector` indicate the primary data structure used in the task
// - <name>: name of the function
// - <mode> defines how the result is returned:
//   - `` default implementation, the user should instanciate the output object before calling the function
//   - `self` writes the result in the same object that is used for input (usually the argument named *_self)
//   - `init` the function instanciates and returns the result object
//
// Functions common to the three data types have helper macros that infers the type at compilation.
// The documentation for these functions is not repeated and only present in the macro definitions.

// ------------------------------------ TASKS FOR DAHL_TENSOR TYPE ------------------------------------
// Sum the tensor values over the t axis and return it as a block of the same x,y shape.
void task_tensor_sum_t_axis(dahl_tensor const* in, dahl_block* out);

// Sum the tensor values over the t axis and initialize + return a block of the same x,y shape.
dahl_block* task_tensor_sum_t_axis_init(dahl_tensor const* in);

// ------------------------------------ TASKS FOR DAHL_BLOCK TYPE ------------------------------------
// Sum the block values over the z axis and return it as a matrix of the same x,y shape.
void task_block_sum_z_axis(dahl_block const* in, dahl_matrix* out);

// Sum the block values over the z axis and initialize + return a matrix of the same x,y shape.
dahl_matrix* task_block_sum_z_axis_init(dahl_block const* in);

dahl_matrix* task_block_sum_y_axis_init(dahl_block const* in);
void task_block_sum_y_axis(dahl_block const* in, dahl_matrix* out);

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
void task_matrix_max_pooling(dahl_matrix const* in, dahl_matrix* out, dahl_matrix* mask, size_t pool_size);

void task_matrix_sum_y_axis(dahl_matrix const* in, dahl_vector* out);
dahl_vector* task_matrix_sum_y_axis_init(dahl_matrix const* in);

// Performs a backward max dahl_pooling, copying each value of `in` into the right index of each window in `out` thanks to the `mask`.
// - `in` shape should be equal to `out` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `out` shape.
void task_matrix_backward_max_pooling(dahl_matrix const* in, dahl_matrix const* mask, dahl_matrix* out, size_t pool_size);

// Same as `task_matrix_backward_max_pooling` but stores the output directly in `mask_self`.
void task_matrix_backward_max_pooling_self(dahl_matrix const* in, dahl_matrix* mask_self, size_t pool_size);

// Performs matrix vector product. Tries to find the right dimension to perform the operation.
void task_matrix_vector_product(dahl_matrix const* mat, dahl_vector const* vec, dahl_vector* out);
dahl_vector* task_matrix_vector_product_init(dahl_matrix const* mat, dahl_vector const* vec);

void task_matrix_matrix_product(dahl_matrix const* a, dahl_matrix const* b, dahl_matrix* c);
dahl_matrix* task_matrix_matrix_product_init(dahl_matrix const* a, dahl_matrix const* b);

void task_matrix_transpose(dahl_matrix const* in, dahl_matrix* out);
dahl_matrix* task_matrix_transpose_init(dahl_matrix const* in);

// Resize a matrix as long as it can hold the same number of elements, e.g. a 4*4 matrix can be resize to 2*8, 1*16...
void task_matrix_resize(dahl_matrix* mat, dahl_shape2d shape);

// Flatten a matrix and consider it as a row matrix
void task_matrix_as_flat_row(dahl_matrix* mat);

// Flatten a matrix and consider it as a column matrix
void task_matrix_as_flat_col(dahl_matrix* mat);

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

void task_vector_softmax_derivative(dahl_vector const* in, dahl_matrix* out);
dahl_matrix* task_vector_softmax_derivative_init(dahl_vector const* in);

dahl_fp task_vector_cross_entropy_loss(dahl_vector const* predictions, dahl_vector const* targets);
dahl_fp task_vector_cross_entropy_loss_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch);

void task_vector_cross_entropy_loss_gradient(dahl_vector const* predictions, dahl_vector const* targets, dahl_vector* gradients);
void task_vector_cross_entropy_loss_gradient_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_matrix* gradient_batch);
dahl_vector* task_vector_cross_entropy_loss_gradient_init(dahl_vector const* predictions, dahl_vector const* targets);
// TODO naming convention
unsigned int task_check_predictions_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch);

// ---------------------------- TASKS FOR ANY TYPES ----------------------------
void task_relu(void const* in, void* out, dahl_traits* traits);
void task_scal(void const* in, void* out, dahl_fp factor, dahl_traits* traits);
void task_power(void const* in, void* out, dahl_fp power, dahl_traits* traits);
void task_sub(void const* a, void const* b, void* c, dahl_traits* traits);
void task_add(void const* a, void const* b, void* c, dahl_traits* traits);
void task_add_value(void const* in, void* out, dahl_fp value, dahl_traits* traits);
void task_clip(void const* in, void* out, dahl_fp min, dahl_fp max, dahl_traits* traits);
dahl_fp task_sum(void const* object, dahl_traits* traits);
void task_fill(void const* object, dahl_fp value, dahl_traits* traits);
void task_wait(void const* object, unsigned int duration, dahl_traits* traits);

// Apply a relu function to every value of `in` and store the result in `out`.
#define TASK_RELU(IN, OUT)                                     \
    do {                                                       \
        _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                       "IN and OUT must be of the same type"); \
        task_relu(IN, OUT, GET_TRAITS(OUT));                   \
    } while (0)

// Update every value of `self` by applying a relu function.
#define TASK_RELU_SELF(SELF) TASK_RELU(SELF, SELF)

// Multiply every value of `in` by `divisor` and store the result in `out`.
#define TASK_SCAL(IN, OUT, FACTOR)                             \
    do {                                                       \
        _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                       "IN and OUT must be of the same type"); \
        task_scal(IN, OUT, FACTOR, GET_TRAITS(OUT));           \
    } while (0)

// Update every value of `self`, multiplying by `divisor`.
#define TASK_SCAL_SELF(SELF, FACTOR) TASK_SCAL(SELF, SELF, FACTOR)

// Power every value of `in` by `power` and store the result in `out`.
#define TASK_POWER(IN, OUT, POWER)                             \
    do {                                                       \
        _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                       "IN and OUT must be of the same type"); \
        task_power(IN, OUT, POWER, GET_TRAITS(OUT));            \
    } while (0)

// Update every value of `self`, powering by `power`.
#define TASK_POWER_SELF(SELF, POWER) TASK_POWER(SELF, SELF, POWER)

// Divide every value of `in` by `divisor` and store the result in `out`.
// Please use scientific notation for the divisor, e.g. 2e0 to divide by two
#define TASK_DIVIDE(IN, OUT, DIVISOR) TASK_SCAL(IN, OUT, 1/(DIVISOR))

// Update every value of `self`, dividing by `divisor`.
// Please use scientific notation for the divisor, e.g. 2e0 to divide by two
#define TASK_DIVIDE_SELF(SELF, DIVISOR) TASK_DIVIDE(SELF, SELF, DIVISOR)

// Performs `c` = `a` - `b`, where:
// - `-` is the value by value substraction
// - `a`, `b` and `c` are dahl_any objects of the same shape
#define TASK_SUB(A, B, C)                               \
    do {                                                \
        _Static_assert(TYPES_MATCH((A), (B)),           \
                   "A and B must be of the same type"); \
        _Static_assert(TYPES_MATCH((A), (C)),           \
                   "A and C must be of the same type"); \
        task_sub(A, B, C, GET_TRAITS(C));               \
    } while (0)

// Performs a_self -= b, where:
// - `-` is the value by value substraction
// - `a_self` and `b` are dahl data structures objects of the same type and shape
// - `a_self` is modified by the function with the substraction result
#define TASK_SUB_SELF(A_SELF, B) TASK_SUB(A_SELF, B, A_SELF)

// Performs `c` = `a` + `b`, where:
// - `+` is the value by value addition
// - `a`, `b` and `c` are dahl_any objects of the same shape
#define TASK_ADD(A, B, C)                               \
    do {                                                \
        _Static_assert(TYPES_MATCH((A), (B)),           \
                   "A and B must be of the same type"); \
        _Static_assert(TYPES_MATCH((A), (C)),           \
                   "A and C must be of the same type"); \
        task_add(A, B, C, GET_TRAITS(C));               \
    } while (0)

// Performs `a_self` += `b`, where:
// - `+` is the value by value addition
// - `a_self` and `b` are dahl_any objects of the same shape
// - `a_self` is modified by the function with the addition result
#define TASK_ADD_SELF(A_SELF, B) TASK_ADD(A_SELF, B, A_SELF)

// Add `value` to every elements of `in` and put the result in `out`
#define TASK_ADD_VALUE(IN, OUT, VALUE)                   \
    do {                                                 \
        _Static_assert(TYPES_MATCH((IN), (OUT)),         \
                   "A and B must be of the same type");  \
        task_add_value(IN, OUT, VALUE, GET_TRAITS(OUT)); \
    } while (0)

// Add `value` to every elements of `self` writing directly in the same buffer
#define TASK_ADD_VALUE_SELF(SELF, VALUE) TASK_ADD_VALUE(SELF, SELF, VALUE)

// Substract `value` to every elements of `in` and put the result in `out`
#define TASK_SUB_VALUE(IN, OUT, VALUE) TASK_ADD_VALUE(IN, OUT, -(VALUE))

// Substract `value` to every elements of `in` writing directly in the same buffer
#define TASK_SUB_VALUE_SELF(SELF, VALUE) TASK_SUB_VALUE(SELF, SELF, VALUE)

// Clip every elements of `in` between `min` and `max` and store to `out`.
#define TASK_CLIP(IN, OUT, MIN, MAX)                    \
    do {                                                \
        _Static_assert(TYPES_MATCH((IN), (OUT)),        \
                   "A and B must be of the same type"); \
        task_clip(IN, OUT, MIN, MAX, GET_TRAITS(OUT));  \
    } while (0)

// Clip and modify every elements of `self` between `min` and `max`.
#define TASK_CLIP_SELF(SELF, MIN, MAX) TASK_CLIP(SELF, SELF, MIN, MAX)

// Sum every elements of `object`.
// No self version of this macro because this is obviously the default behavior.
#define TASK_SUM(OBJECT) task_sum(OBJECT, GET_TRAITS(OBJECT))

// Fill every elements of `object` with `value`.
// No self version of this macro because this is obviously the default behavior.
#define TASK_FILL(OBJECT, VALUE) task_fill(OBJECT, VALUE, GET_TRAITS(OBJECT))

// Acquire `object` and wait for `duration` in microseconds.
// Useful for debug purposes to investigate possible synchronization issues.
#define TASK_WAIT(OBJECT, DURATION) task_wait(OBJECT, DURATION, GET_TRAITS(OBJECT))

#endif //!DAHL_TASKS_H
