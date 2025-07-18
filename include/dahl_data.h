#ifndef DAHL_DATA_H
#define DAHL_DATA_H

#include "dahl_types.h"
#include "dahl_arena.h"

typedef struct _dahl_tensor dahl_tensor;
typedef struct _dahl_block dahl_block;
typedef struct _dahl_matrix dahl_matrix;
typedef struct _dahl_vector dahl_vector;

typedef struct _dahl_partition dahl_partition;

// Using a trait mechanism to group functions together
typedef const struct _dahl_traits dahl_traits;

// Here we define our different traits associated to each types.
// Those structures contains references to the implementation of each function matching with the correct type.
extern dahl_traits dahl_traits_tensor;
extern dahl_traits dahl_traits_block;
extern dahl_traits dahl_traits_matrix;
extern dahl_traits dahl_traits_vector;

// Get the traits structure of an object at compile time
#define GET_TRAITS(OBJECT) _Generic((OBJECT), \
    dahl_tensor*: &dahl_traits_tensor,        \
    dahl_block*:  &dahl_traits_block,         \
    dahl_matrix*: &dahl_traits_matrix,        \
    dahl_vector*: &dahl_traits_vector,        \
    dahl_tensor const*: &dahl_traits_tensor,  \
    dahl_block const*:  &dahl_traits_block,   \
    dahl_matrix const*: &dahl_traits_matrix,  \
    dahl_vector const*: &dahl_traits_vector   \
)

// Type comparison without taking into account const qualifiers
#define TYPES_MATCH(T1, T2) \
    (__builtin_types_compatible_p(typeof(*(T1)), typeof(*(T2))))

// ---------------------------------------- TENSOR ----------------------------------------
// Initialize a dahl_tensor with every values at 0.
// parameters:
// - shape: dahl_shape4d object describing the dimensions of the tensor
dahl_tensor* tensor_init(dahl_arena*, dahl_shape4d shape);

// Initialize a dahl_tensor with random values.
// parameters:
// - shape: dahl_shape4d object describing the dimensions of the tensor
dahl_tensor* tensor_init_random(dahl_arena*, dahl_shape4d shape);

// Initialize a dahl_tensor by cloning an existing array.
// Cloned memory will be freed upon calling `tensor_finalize`, however do not forget to free the original array.
// - shape: dahl_shape4d object describing the dimensions of the tensor
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_tensor* tensor_init_from(dahl_arena*, dahl_shape4d shape, dahl_fp const* data);

// Clone a tensor
dahl_tensor* tensor_clone(dahl_arena*, dahl_tensor const*);

// Returns the tensor shape
dahl_shape4d tensor_get_shape(dahl_tensor const*);

// Compares two tensors value by value and returns wether or not they're equal.
bool tensor_equals(dahl_tensor const* a, dahl_tensor const* b, bool rounding, u_int8_t precision);

// Acquire the tensor data, will wait any associated tasks to finish.
dahl_fp const* tensor_data_acquire(dahl_tensor const*);

// Acquire the tensor data as mut, will wait any associated tasks to finish.
// Caution: will cause dead lock if the data is already partitionned.
dahl_fp* tensor_data_acquire_mut(dahl_tensor*);

// Release the tensor data, tasks will be able to use the tensor again.
void tensor_data_release(dahl_tensor const*);

// Partition data along z axis, the sub matrices can then be accesed with `GET_SUB_MATRIX`.
// Exactly creates z sub matrices, so `GET_NB_CHILDREN` should be equal to z.
// Note the the tensor itself cannot be used as long as it is partitioned.
void tensor_partition_along_t(dahl_tensor const*);
void tensor_partition_along_t_mut(dahl_tensor*);

// Unpartition a tensor
void tensor_unpartition(dahl_tensor const*);

// Print a tensor
void tensor_print(dahl_tensor const*);

// ---------------------------------------- BLOCK ----------------------------------------
// Initialize a dahl_block with every values at 0.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init(dahl_arena*, dahl_shape3d shape);

// Initialize a dahl_block with random values.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init_random(dahl_arena*, dahl_shape3d shape);

// Initialize a dahl_block by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape3d object describing the dimensions of the block
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_block* block_init_from(dahl_arena*, dahl_shape3d shape, dahl_fp const* data);

// Clone a block
dahl_block* block_clone(dahl_arena*, dahl_block const*);

// Returns a new block with added padding. 
// The new_shape should be larger than the previous block.
// If it is exactly the same, it just produces a copy of the bolck.
// If the new padding is even, the remainder is placed at the end of the axis.
dahl_block* block_add_padding_init(dahl_arena*, dahl_block const*, dahl_shape3d new_shape);

// Returns the block shape
dahl_shape3d block_get_shape(dahl_block const*);

// Compares two blocks value by value and returns wether or not they're equal.
bool block_equals(dahl_block const* a, dahl_block const* b, bool rounding, u_int8_t precision);

// Acquire the block data, will wait any associated tasks to finish.
dahl_fp const* block_data_acquire(dahl_block const*);

// Acquire the block data as mut, will wait any associated tasks to finish.
// Caution: will cause dead lock if the data is already partitionned.
dahl_fp* block_data_acquire_mut(dahl_block*);

// Release the block data, tasks will be able to use the block again.
void block_data_release(dahl_block const*);

// Partition data along z axis, the sub matrices can then be accesed with `GET_SUB_MATRIX`.
// Exactly creates z sub matrices, so `GET_NB_CHILDREN` should be equal to z.
// Note the the block itself cannot be used as long as it is partitioned.
void block_partition_along_z(dahl_block const*);
void block_partition_along_z_mut(dahl_block*);

// Same that `block_partition_along_z` but actually produces flattened matrices of the matrices on x,y.
// Chose wether the flattened matrices are row matrices or column matrices with `is_row`.
// Exactly creates z sub vectors, so `GET_NB_CHILDREN` should be equal to z.
void block_partition_along_z_flat_matrices(dahl_block const*, bool is_row);
void block_partition_along_z_flat_matrices_mut(dahl_block*, bool is_row);

// Same that `block_partition_along_z` but actually produces flattened vectors of the matrices on x,y.
// Exactly creates z sub vectors, so `GET_NB_CHILDREN` should be equal to z.
void block_partition_along_z_flat_vectors(dahl_block const*);
void block_partition_along_z_flat_vectors_mut(dahl_block*);

void block_partition_flatten_to_vector(dahl_block const*);
void block_partition_flatten_to_vector_mut(dahl_block*);

// Partition along z but by batch, so it creates sub blocks.
// TODO: support or return an error if the batch size does not divide properly the block
void block_partition_along_z_batch(dahl_block const*, size_t batch_size);
void block_partition_along_z_batch_mut(dahl_block*, size_t batch_size);

// Unpartition a block
void block_unpartition(dahl_block const*);

// Print a block
void block_print(dahl_block const*);

// ---------------------------------------- MATRIX ----------------------------------------
// Initialize a dahl_matrix with every values at 0.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init(dahl_arena*, dahl_shape2d shape);

// Initialize a dahl_matrix with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init_random(dahl_arena*, dahl_shape2d shape);

// Initialize a dahl_matrix by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the matrix
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_matrix* matrix_init_from(dahl_arena*, dahl_shape2d shape, dahl_fp const* data);

// Clone a matrix
dahl_matrix* matrix_clone(dahl_arena*, dahl_matrix const*);

// Returns the matrix shape
dahl_shape2d matrix_get_shape(dahl_matrix const*);

// Acquire the matrix data, will wait any associated tasks to finish.
dahl_fp const* matrix_data_acquire(dahl_matrix const*);

// Acquire the matrix data as mut, will wait any associated tasks to finish.
// Caution: will cause dead lock if the data is already partitionned.
dahl_fp* matrix_data_acquire_mut(dahl_matrix*);

// Release the matrix data, tasks will be able to use the block again.
void matrix_data_release(dahl_matrix const*);

// Compares two matrices value by value and returns wether or not they're equal.
bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool rounding, u_int8_t precision);

// Partition data along y axis, the sub vectors can then be accesed with `GET_SUB_VECTOR`.
// Exactly creates y sub vectors, so `GET_NB_CHILDREN` should be equal to y.
// Note the the vector itself cannot be used as long as it is partitioned.
void matrix_partition_along_y(dahl_matrix const*);

// Same as `matrix_partition_along_y` but mutably acquires the data.
void matrix_partition_along_y_mut(dahl_matrix*);

// Partition data along y axis, and produces sub matrices of shape (x, y / batch_size).
// Exactly creates y / batch_size sub matrices that can be accessed with `GET_SUB_MATRIX`.
// Note the the vector itself cannot be used as long as it is partitioned.
// TODO: support or return an error if the batch size does not divide properly the matrix
void matrix_partition_along_y_batch(dahl_matrix const*, size_t batch_size);

// Same as `matrix_partition_along_y_batch` but mutably acquires the data.
void matrix_partition_along_y_batch_mut(dahl_matrix*, size_t batch_size);

// Unpartition a matrix
void matrix_unpartition(dahl_matrix const*);

// Print a matrix
void matrix_print(dahl_matrix const*);

// Print a matrix with ascii format, useful to print images in the terminal
void matrix_print_ascii(dahl_matrix const*, dahl_fp threshold);

// ---------------------------------------- VECTOR ----------------------------------------
// Initialize a dahl_vector with every values at 0.
// parameters:
// - len: size_t lenght of the vector
dahl_vector* vector_init(dahl_arena*, size_t len);

// Initialize a dahl_vector with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the vector
dahl_vector* vector_init_random(dahl_arena*, size_t len);

// Initialize a dahl_vector by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the vector
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_vector* vector_init_from(dahl_arena*, size_t len, dahl_fp const* data);

// Clone a vector
dahl_vector* vector_clone(dahl_arena*, dahl_vector const*);

// Returns the vector len
size_t vector_get_len(dahl_vector const*);

// Acquire the vector data, will wait any associated tasks to finish.
dahl_fp const* vector_data_acquire(dahl_vector const*);

// Acquire the vector data as mut, will wait any associated tasks to finish.
// Caution: will cause dead lock if the data is already partitionned.
dahl_fp* vector_data_acquire_mut(dahl_vector*);

// Release the vector data, tasks will be able to use the block again.
void vector_data_release(dahl_vector const*);

// Copy the vector into a new categorical matrix
// E.g. [1,2,0,1,1] gives:
// [[0, 1, 0],
//  [0, 0, 1],
//  [1, 0, 0],
//  [0, 1, 0],
//  [0, 1, 0]]
dahl_matrix* vector_to_categorical(dahl_arena*, dahl_vector const*, size_t num_classes);

// Compares the two matrices value by value and returns wether or not they're equal.
// Rounding values can be enabled or disabled, and rounding precision can be specified.
bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool rounding, u_int8_t precision);

// Print a vector
void vector_print(dahl_vector const*);

// ---------------------------------------- PARTITION ----------------------------------------

size_t get_nb_children(void const* object, dahl_traits* traits);

dahl_block const* get_sub_block(void const* object, size_t index, dahl_traits* traits);
dahl_block* get_sub_block_mut(void* object, size_t index, dahl_traits* traits);

dahl_matrix const* get_sub_matrix(void const* object, size_t index, dahl_traits* traits);
dahl_matrix* get_sub_matrix_mut(void* object, size_t index, dahl_traits* traits);

dahl_vector const* get_sub_vector(void const* object, size_t index, dahl_traits* traits);
dahl_vector* get_sub_vector_mut(void* object, size_t index, dahl_traits* traits);

#define GET_NB_CHILDREN(OBJECT) get_nb_children(OBJECT, GET_TRAITS(OBJECT))

#define GET_SUB_BLOCK(OBJECT, INDEX) get_sub_block(OBJECT, INDEX, GET_TRAITS(OBJECT))
#define GET_SUB_BLOCK_MUT(OBJECT, INDEX) get_sub_block_mut(OBJECT, INDEX, GET_TRAITS(OBJECT))

#define GET_SUB_MATRIX(OBJECT, INDEX) get_sub_matrix(OBJECT, INDEX, GET_TRAITS(OBJECT))
#define GET_SUB_MATRIX_MUT(OBJECT, INDEX) get_sub_matrix_mut(OBJECT, INDEX, GET_TRAITS(OBJECT))

#define GET_SUB_VECTOR(OBJECT, INDEX) get_sub_vector(OBJECT, INDEX, GET_TRAITS(OBJECT))
#define GET_SUB_VECTOR_MUT(OBJECT, INDEX) get_sub_vector_mut(OBJECT, INDEX, GET_TRAITS(OBJECT))


#endif //!DAHL_DATA_H
