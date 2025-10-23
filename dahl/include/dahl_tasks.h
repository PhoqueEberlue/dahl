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
dahl_block* task_tensor_sum_t_axis_init(dahl_arena*, dahl_tensor const* in);

void task_tensor_sum_xyt_axes(dahl_tensor const* in, dahl_vector* out);
dahl_vector* task_tensor_sum_xyt_axes_init(dahl_arena*, dahl_tensor const* in);

// ------------------------------------ TASKS FOR DAHL_BLOCK TYPE ------------------------------------
// Sum the block values over the z axis and return it as a matrix of the same x,y shape.
void task_block_sum_z_axis(dahl_block const* in, dahl_matrix* out);
dahl_matrix* task_block_sum_z_axis_init(dahl_arena*, dahl_block const* in);

// Sum the block values over the y axis and return it as a matrix of the same x,z shape.
void task_block_sum_y_axis(dahl_block const* in, dahl_matrix* out);
dahl_matrix* task_block_sum_y_axis_init(dahl_arena*, dahl_block const* in);

// Sum the block values over the x and y axes and store the result in the vector `out` of len z.
// `out` is compatible with redux objects.
void task_block_sum_xy_axes(dahl_block const* in, dahl_vector* out);

// Sum the block values over the x and y axes and return a new vector `out` of len z.
dahl_vector* task_block_sum_xy_axes_init(dahl_arena*, dahl_block const* in);

// Add padding to the block `in` by fitting it to `out` that should be of a greater shape.
// E.g.: in(4,3,2) with out(6,5,2) will add zeros on the corner of the matrices, but not on the z dimension.
// If the new padding is even, the remainder is placed at the end of the axis.
void task_block_add_padding(dahl_block const* in, dahl_block* out);

// Add padding to the block `in` by returning a new block with the `new_shape` that should be greater than `in`'s shape.
// If the new padding is even, the remainder is placed at the end of the axis.
dahl_block* task_block_add_padding_init(dahl_arena*, dahl_block const* in, dahl_shape3d new_shape);

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
void task_matrix_max_pooling(dahl_matrix const* in, dahl_matrix* mask, dahl_matrix* out, size_t pool_size);

// Sum the matrix values over the y axis and return it as a vector of len x.
void task_matrix_sum_y_axis(dahl_matrix const* in, dahl_vector* out);
dahl_vector* task_matrix_sum_y_axis_init(dahl_arena*, dahl_matrix const* in);

// Performs a backward max dahl_pooling, copying each value of `in` into the right index of each window in `out` thanks to the `mask`.
// - `in` shape should be equal to `out` shape / `pool_size` (euclidian division)
// - `mask` shape should be the same as `out` shape.
void task_matrix_backward_max_pooling(dahl_matrix const* in, dahl_matrix const* mask, dahl_matrix* out, size_t pool_size);

// Same as `task_matrix_backward_max_pooling` but stores the output directly in `mask_self`.
void task_matrix_backward_max_pooling_self(dahl_matrix const* in, dahl_matrix* mask_self, size_t pool_size);

// Performs matrix vector product. Tries to find the right dimension to perform the operation.
void task_matrix_vector_product(dahl_matrix const* mat, dahl_vector const* vec, dahl_vector* out);
dahl_vector* task_matrix_vector_product_init(dahl_arena*, dahl_matrix const* mat, dahl_vector const* vec);

void task_matrix_matrix_product(dahl_matrix const* a, dahl_matrix const* b, dahl_matrix* c);
dahl_matrix* task_matrix_matrix_product_init(dahl_arena*, dahl_matrix const* a, dahl_matrix const* b);

void task_matrix_transpose(dahl_matrix const* in, dahl_matrix* out);
dahl_matrix* task_matrix_transpose_init(dahl_arena*, dahl_matrix const* in);

// Resize a matrix as long as it can hold the same number of elements, e.g. a 4*4 matrix can be resize to 2*8, 1*16...
void task_matrix_resize(dahl_matrix*, dahl_shape2d shape);

// Flatten a matrix and consider it as a row matrix
void task_matrix_as_flat_row(dahl_matrix*);

// Flatten a matrix and consider it as a column matrix
void task_matrix_as_flat_col(dahl_matrix*);

// Rotate matrix `in` by 180 degrees and store result into `out`
void task_matrix_rotate_180(dahl_matrix const* in, dahl_matrix* out);

// Rotate matrix `in` by 180 degrees and return the result into a new matrix
dahl_matrix* task_matrix_rotate_180_init(dahl_arena* arena, dahl_matrix const* in);

// ------------------------------------ TASKS FOR DAHL_VECTOR TYPE ------------------------------------
// Performs the softmax function with `in` vector and writes the result to `out`.
void task_vector_softmax(dahl_vector const* in, dahl_vector* out);
dahl_vector* task_vector_softmax_init(dahl_arena*, dahl_vector const* in);

// Performs `a`  `b` = `c`, where:
// - `` is the dot product
// - `a`, `b` are dahl_vector of the same length
// - `c` is a scalar containing the result
void task_vector_dot_product(dahl_vector const* a, dahl_vector const* b, dahl_scalar* c);
dahl_scalar* task_vector_dot_product_init(dahl_arena* arena, dahl_vector const* a, dahl_vector const* b);

void task_vector_diag(dahl_vector const* in, dahl_matrix* out);
// Create and return a diagonal dahl_matrix of the input dahl_vector
dahl_matrix* task_vector_diag_init(dahl_arena*, dahl_vector const* in);

// Performs the softmax derivative on `in` and store into `out`. 
// The `scratch_arena` is used to store partial results of this task.
void task_vector_softmax_derivative(dahl_arena* scratch_arena, dahl_vector const* in, dahl_matrix* out);
dahl_matrix* task_vector_softmax_derivative_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_vector const* in);

void task_vector_to_matrix(dahl_vector const* in, dahl_matrix* out);

// Copy the vector into a new matrix. The shape product must be equal to the lenght of the orignal vector (x*y==len)
dahl_matrix* task_vector_to_matrix_init(dahl_arena*, dahl_vector const*, dahl_shape2d new_shape);

// Copy the vector into a new column matrix of shape (1, len)
dahl_matrix* task_vector_to_column_matrix_init(dahl_arena*, dahl_vector const*);

// Copy the vector into a new row matrix of shape (len, 1)
dahl_matrix* task_vector_to_row_matrix_init(dahl_arena*, dahl_vector const*);

// Compute the outer product of vectors `a` and `b`, storing the result into the matrix `c` with dimensions len(a) x len(b)
// `c` is compatible with redux objects.
void task_vector_outer_product(dahl_vector const* a, dahl_vector const* b, dahl_matrix* c);

// Compute the outer product of vectors `a` and `b`, returning the result into a new matrix with dimensions len(a) x len(b)
dahl_matrix* task_vector_outer_product_init(dahl_arena* arena, dahl_vector const* a, dahl_vector const* b);

// Shuffles directly `vec`.
void task_vector_shuffle(dahl_vector* vec);

// Performs vector matrix product. This is different than matrix vector product because it places
// the vector as the first argument of the operation, giving different dimension as result:
// The length of `vec` should be equal to dimension `mat_y`, and the output will be of len mat_x
void task_vector_matrix_product(dahl_vector const* vec, dahl_matrix const* mat, dahl_vector* out);
dahl_vector* task_vector_matrix_product_init(dahl_arena*, dahl_vector const* vec, dahl_matrix const* mat);

// ---------------------------- TASKS FOR ANY TYPES ----------------------------
void task_relu(void const* in, void* out, dahl_traits* traits);
void task_relu_backward(void const* input, void const* gradients, void* out, dahl_traits* traits);
void task_scal(void const* in, void* out, dahl_fp factor, dahl_traits* traits);
void task_power(void const* in, void* out, dahl_fp power, dahl_traits* traits);
void task_sub(void const* a, void const* b, void* c, dahl_traits* traits);
void task_add(void const* a, void const* b, void* c, dahl_traits* traits);
void task_add_value(void const* in, void* out, dahl_fp value, dahl_traits* traits);
void task_clip(void const* in, void* out, dahl_fp min, dahl_fp max, dahl_traits* traits);
void task_sum(void const* in, dahl_scalar* out, dahl_traits* traits);
dahl_scalar* task_sum_init(dahl_arena*, void const* object, dahl_traits* traits);
void task_mean(void const* in, dahl_scalar* out, dahl_traits* traits);
dahl_scalar* task_mean_init(dahl_arena*, void const* object, dahl_traits* traits);
void task_fill(void* object, dahl_fp value, dahl_traits* traits);
void task_wait(void const* object, unsigned int duration, dahl_traits* traits);
void task_copy(void const* in, void* out, dahl_traits* traits);
void task_min(void const* in, dahl_scalar* out, dahl_traits* traits);
dahl_scalar* task_min_init(dahl_arena*, void const* object, dahl_traits* traits);
void task_max(void const* in, dahl_scalar* out, dahl_traits* traits);
dahl_scalar* task_max_init(dahl_arena*, void const* object, dahl_traits* traits);
void task_round(void const* in, void* out, int8_t precision, dahl_traits* traits);

// Apply a relu function to every value of `in` and store the result in `out`.
#define TASK_RELU(IN, OUT)                                     \
    do {                                                       \
        _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                       "IN and OUT must be of the same type"); \
        task_relu(IN, OUT, GET_TRAITS(OUT));                   \
    } while (0)

// Update every value of `self` by applying a relu function.
#define TASK_RELU_SELF(SELF) TASK_RELU(SELF, SELF)

// Apply backward relu function to every value of `in` and store the result in `out`.
#define TASK_RELU_BACKWARD(INPUT, GRADIENTS, OUT)                   \
    do {                                                            \
        _Static_assert(TYPES_MATCH((INPUT), (GRADIENTS)),           \
                       "IN and OUT must be of the same type");      \
        _Static_assert(TYPES_MATCH((INPUT), (OUT)),                 \
                       "IN and OUT must be of the same type");      \
        task_relu_backward(INPUT, GRADIENTS, OUT, GET_TRAITS(OUT)); \
    } while (0)

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
// `c` is compatible with redux objects.
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
// /!\ Care: `c` is NOT compatible with redux objects on the self version.
// TODO Find a way to make it work
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

// Sum every elements of `in` and write the result into a scalar `out`. 
// `out` is compatible with redux objects.
#define TASK_SUM(IN, OUT) task_sum(IN, OUT, GET_TRAITS(IN))

// Sum every elements of `in` and allocate/return the result into the `arena`.
#define TASK_SUM_INIT(ARENA, OBJECT) task_sum_init(ARENA, OBJECT, GET_TRAITS(OBJECT))

// Compute mean for every elements of `in` and write the result into a scalar `out`.
#define TASK_MEAN(IN, OUT) task_mean(IN, OUT, GET_TRAITS(OBJECT))

// Compute mean for every elements of `in` and allocate/return the result into the `arena`.
#define TASK_MEAN_INIT(ARENA, OBJECT) task_mean_init(ARENA, OBJECT, GET_TRAITS(OBJECT))

// Fill every elements of `object` with `value`.
// No self version of this macro because this is obviously the default behavior.
#define TASK_FILL(OBJECT, VALUE) task_fill(OBJECT, VALUE, GET_TRAITS(OBJECT))

// Acquire `object` and wait for `duration` in microseconds.
// Useful for debug purposes to investigate possible synchronization issues.
#define TASK_WAIT(OBJECT, DURATION) task_wait(OBJECT, DURATION, GET_TRAITS(OBJECT))

// Copy `in` values into `out`.
#define TASK_COPY(IN, OUT)                              \
    do {                                                \
        _Static_assert(TYPES_MATCH((IN), (OUT)),        \
                   "A and B must be of the same type"); \
        task_copy(IN, OUT, GET_TRAITS(OUT));            \
    } while (0)

// Find the minimum value of `in` and write the result into a scalar `out`.
#define TASK_MIN(IN, OUT) task_min(IN, OUT, GET_TRAITS(OBJECT))

// Find the minimum value of `in` and allocate/return the result into the `arena`.
#define TASK_MIN_INIT(ARENA, OBJECT) task_min_init(ARENA, OBJECT, GET_TRAITS(OBJECT))

// Find the maximum value of `in` and write the result into a scalar `out`.
#define TASK_MAX(IN, OUT) task_max(IN, OUT, GET_TRAITS(OBJECT))

// Find the maximum value of `in` and allocate/return the result into the `arena`.
#define TASK_MAX_INIT(ARENA, OBJECT) task_max_init(ARENA, OBJECT, GET_TRAITS(OBJECT))

// Apply a round function to every value of `in` with `precision` and store the result in `out`.
#define TASK_ROUND(IN, OUT, PRECISION)                         \
    do {                                                       \
        _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                       "IN and OUT must be of the same type"); \
        task_round(IN, OUT, PRECISION, GET_TRAITS(OUT));       \
    } while (0)

// Apply a round function to every value of `in` with `precision` and store the result in `out`.
#define TASK_ROUND_SELF(SELF, PRECISION) TASK_ROUND(SELF, SELF, PRECISION)

// ---------------------------- VARIOUS TASKS RELATED TO ML ----------------------------
// Count the number of good predictions from the `prediction_batch` and `target_batch`.
void task_check_predictions_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* good_predictions);

// Init and return the number of good predictions from the `prediction_batch` and `target_batch`.
dahl_scalar* task_check_predictions_batch_init(dahl_arena* arena, dahl_matrix const* prediction_batch, dahl_matrix const* target_batch);

// Compute the cross entropy loss over the given batch and writes the result into `out`.
void task_cross_entropy_loss_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* out);

// Compute the cross entropy loss over the given batch and return the result into a new scalar.
dahl_scalar* task_cross_entropy_loss_batch_init(dahl_arena* arena, dahl_matrix const* prediction_batch, dahl_matrix const* target_batch);

void task_cross_entropy_loss_gradient_batch(dahl_matrix const* predictions, dahl_matrix const* targets, dahl_matrix* gradients);

dahl_matrix* task_cross_entropy_loss_gradient_batch_init(dahl_arena* arena, dahl_matrix const* prediction_batch, 
                                                                dahl_matrix const* target_batch);
// Performs `out` = `in` x `kernel`, where:
// - x is the cross correlation operator over multiple channels
// - `in`, `kernel` and `out` are dahl_block objects
// - the shape of `out` must respect: out_nx = in_nx - kernel_nx + 1 and out_ny = in_ny - kernel_ny + 1. No restriction on Z axis because its the channel dimension and the results gets accumulated into `out` which is a matrix.
// - the shape of the `kernel` should be smaller than the shape of `in` 
void task_convolution_2d(dahl_block const* in, dahl_block const* kernel, dahl_matrix* out);

// Specialized convolution 2d function to compute dl_dfilters. 
// It accepts the orginal forward input for argument `in` which can be a block (image with multiple channels).
// `kernel` is here the filters gradients represented as a matrix.
// `out` is derivative of the filters with respect to the input.
// Results are computed on each `in` channels and stored on their respective `out` channel.
// `out` is compatible with redux objects.
void task_convolution_2d_backward_filters(dahl_block const* in, dahl_matrix const* kernel, dahl_block* out);

// Specialized convolution 2d function to compute dl_dinput. 
// It accepts the gradients with a matrix shape for `in`.
// `kernel` is here the filters associated to the convolution represented as a block.
// `out` is derivative of the input.
// Results are computed on each `kernel` channels and stored on their respective `out` channel.
// `out` is compatible with redux objects.
void task_convolution_2d_backward_input(dahl_matrix const* in, dahl_block const* kernel, dahl_block* out);

void task_convolution_2d_backward_input_padding_free(dahl_matrix const* in, dahl_block const* kernel, dahl_block* out);

#endif //!DAHL_TASKS_H
