#ifndef DAHL_DATA_H
#define DAHL_DATA_H

#include "dahl_types.h"

typedef struct _dahl_tensor dahl_tensor;
typedef struct _dahl_block dahl_block;
typedef struct _dahl_matrix dahl_matrix;
typedef struct _dahl_vector dahl_vector;

// ---------------------------------------- TENSOR ----------------------------------------
// Initialize a dahl_tensor with every values at 0.
// parameters:
// - shape: dahl_shape4d object describing the dimensions of the tensor
dahl_tensor* tensor_init(dahl_shape4d shape);

// Initialize a dahl_tensor with random values.
// parameters:
// - shape: dahl_shape4d object describing the dimensions of the tensor
dahl_tensor* tensor_init_random(dahl_shape4d shape);

// Initialize a dahl_tensor by cloning an existing array.
// Cloned memory will be freed upon calling `tensor_finalize`, however do not forget to free the original array.
// - shape: dahl_shape4d object describing the dimensions of the tensor
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_tensor* tensor_init_from(dahl_shape4d shape, dahl_fp const* data);

// Clone a tensor
dahl_tensor* tensor_clone(dahl_tensor const* tensor);

// Returns a new tensor with added padding. 
// The new_shape should be larger than the previous tensor.
// If it is exactly the same, it just produces a copy of the bolck.
// If the new padding is even, the remainder is placed at the end of the axis.
dahl_tensor* tensor_add_padding_init(dahl_tensor const* tensor, dahl_shape4d new_shape);

// Returns the tensor shape
dahl_shape4d tensor_get_shape(dahl_tensor const* tensor);

// Compares two tensors value by value and returns wether or not they're equal.
bool tensor_equals(dahl_tensor const* a, dahl_tensor const* b, bool rounding, u_int8_t precision);

// Acquire the tensor data, will wait any associated tasks to finish.
dahl_fp* tensor_data_acquire(dahl_tensor const* tensor);

// Release the tensor data, tasks will be able to use the tensor again.
void tensor_data_release(dahl_tensor const* tensor);

// Partition data along z axis, the sub matrices can then be accesed with `tensor_get_sub_matrix`.
// Exactly creates z sub matrices, so `tensor_get_sub_matrix_nb` should be equal to z.
// Note the the tensor itself cannot be used as long as it is partitioned.
void tensor_partition_along_t(dahl_tensor* tensor);

// Unpartition a tensor
void tensor_unpartition(dahl_tensor* tensor);

// Get the number of children
size_t tensor_get_nb_children(dahl_tensor const* tensor);

// Get sub block at `index`. To be called after `tensor_partition_along_t`.
dahl_block* tensor_get_sub_block(dahl_tensor const* tensor, size_t index);

// Get sub matrix at `index`. To be called after `tensor_partition_along_z`.
dahl_matrix* tensor_get_sub_matrix(dahl_tensor const* tensor, size_t index);

// Get sub vector at `index`. To be called after `tensor_partition_along_z_flat`.
dahl_vector* tensor_get_sub_vector(dahl_tensor const* tensor, size_t index);

// Print a tensor
void tensor_print(dahl_tensor const* tensor);

// ---------------------------------------- BLOCK ----------------------------------------
// Initialize a dahl_block with every values at 0.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init(dahl_shape3d shape);

// Initialize a dahl_block with random values.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init_random(dahl_shape3d shape);

// Initialize a dahl_block by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape3d object describing the dimensions of the block
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_block* block_init_from(dahl_shape3d shape, dahl_fp const* data);

// Clone a block
dahl_block* block_clone(dahl_block const* block);

// Returns a new block with added padding. 
// The new_shape should be larger than the previous block.
// If it is exactly the same, it just produces a copy of the bolck.
// If the new padding is even, the remainder is placed at the end of the axis.
dahl_block* block_add_padding_init(dahl_block const* block, dahl_shape3d new_shape);

// Returns the block shape
dahl_shape3d block_get_shape(dahl_block const* block);

// Compares two blocks value by value and returns wether or not they're equal.
bool block_equals(dahl_block const* a, dahl_block const* b, bool rounding, u_int8_t precision);

// Acquire the block data, will wait any associated tasks to finish.
dahl_fp* block_data_acquire(dahl_block const* block);

// Release the block data, tasks will be able to use the block again.
void block_data_release(dahl_block const* block);

// Partition data along z axis, the sub matrices can then be accesed with `block_get_sub_matrix`.
// Exactly creates z sub matrices, so `block_get_nb_children` should be equal to z.
// Note the the block itself cannot be used as long as it is partitioned.
void block_partition_along_z(dahl_block* block);

// Same that `block_partition_along_z` but actually produces flattened vectors of the matrices on x,y.
// Exactly creates z sub vectors, so `block_get_sub_vector_nb` should be equal to z.
void block_partition_along_z_flat(dahl_block* block);

void block_partition_flatten_to_vector(dahl_block* block);

// Unpartition a block
void block_unpartition(dahl_block* block);

// Get the number of children
size_t block_get_nb_children(dahl_block const* block);

// Get sub matrix at index. To be called after `block_partition_along_z`.
dahl_matrix* block_get_sub_matrix(dahl_block const* block, size_t index);

// Get sub vector at index. To be called after `block_partition_along_z_flat`.
dahl_vector* block_get_sub_vector(dahl_block const* block, size_t index);

// Print a block
void block_print(dahl_block const* block);

// ---------------------------------------- MATRIX ----------------------------------------
// Initialize a dahl_matrix with every values at 0.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init(dahl_shape2d shape);

// Initialize a dahl_matrix with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init_random(dahl_shape2d shape);

// Initialize a dahl_matrix by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the matrix
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_matrix* matrix_init_from(dahl_shape2d shape, dahl_fp const* data);

// Clone a matrix
dahl_matrix* matrix_clone(dahl_matrix const* matrix);

// Returns the matrix shape
dahl_shape2d matrix_get_shape(dahl_matrix const* matrix);

// Acquire the matrix data, will wait any associated tasks to finish.
dahl_fp* matrix_data_acquire(dahl_matrix const* matrix);

// Release the matrix data, tasks will be able to use the block again.
void matrix_data_release(dahl_matrix const* matrix);

// Compares two matrices value by value and returns wether or not they're equal.
bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool rounding, u_int8_t precision);

// Partition data along y axis, the sub vectors can then be accesed with `matrix_get_sub_vector`.
// Exactly creates y sub vectors, so `matrix_get_nb_children` should be equal to y.
// Note the the vector itself cannot be used as long as it is partitioned.
void matrix_partition_along_y(dahl_matrix* matrix);

// Unpartition a matrix
void matrix_unpartition(dahl_matrix* matrix);

// Get the number of children
size_t matrix_get_nb_children(dahl_matrix const* matrix);

// Get the sub vector at index
dahl_vector* matrix_get_sub_vector(dahl_matrix const* matrix, size_t index);

// Print a matrix
void matrix_print(dahl_matrix const* matrix);

// Print a matrix with ascii format, useful to print images in the terminal
void matrix_print_ascii(dahl_matrix const* matrix, dahl_fp threshold);

// ---------------------------------------- VECTOR ----------------------------------------
// Initialize a dahl_vector with every values at 0.
// parameters:
// - len: size_t lenght of the vector
dahl_vector* vector_init(size_t len);

// Initialize a dahl_vector with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the vector
dahl_vector* vector_init_random(size_t len);

// Initialize a dahl_vector by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the vector
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_vector* vector_init_from(size_t len, dahl_fp const* data);

// Clone a vector
dahl_vector* vector_clone(dahl_vector const* vector);

// Returns the vector len
size_t vector_get_len(dahl_vector const* vector);

// Acquire the vector data, will wait any associated tasks to finish.
dahl_fp* vector_data_acquire(dahl_vector const* vector);

// Release the vector data, tasks will be able to use the block again.
void vector_data_release(dahl_vector const* vector);

// Copy the vector into a new matrix. The shape product must be equal to the lenght of the orignal vector (x*y==len)
dahl_matrix* vector_to_matrix(dahl_vector const* vector, dahl_shape2d shape);

// Copy the vector into a new column matrix of shape (1, len)
dahl_matrix* vector_to_column_matrix(dahl_vector const* vector);

// Copy the vector into a new row matrix of shape (len, 1)
dahl_matrix* vector_to_row_matrix(dahl_vector const* vector);

// Copy the vector into a new categorical matrix
// E.g. [1,2,0,1,1] gives:
// [[0, 1, 0],
//  [0, 0, 1],
//  [1, 0, 0],
//  [0, 1, 0],
//  [0, 1, 0]]
dahl_matrix* vector_to_categorical(dahl_vector const* vector, size_t num_classes);

// Compares the two matrices value by value and returns wether or not they're equal.
// Rounding values can be enabled or disabled, and rounding precision can be specified.
bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool rounding, u_int8_t precision);

// Print a vector
void vector_print(dahl_vector const* vector);

#endif //!DAHL_DATA_H
