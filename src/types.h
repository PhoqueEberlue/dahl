#ifndef DAHL_TYPES_H
#define DAHL_TYPES_H

#include <starpu.h>
#include <stdlib.h>

typedef double dahl_fp;

typedef struct
{
    size_t x;
    size_t y;
    size_t z;
} shape3d;

typedef struct
{
    size_t x;
    size_t y;
} shape2d;

typedef struct
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    // Wether this vector is a sub matrix data
    bool is_sub_matrix_data;
} dahl_vector;

typedef struct
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    dahl_vector* sub_vectors;
    bool is_partitioned;

    // Wether this matrix is a sub block data
    bool is_sub_block_data;
} dahl_matrix;

typedef struct
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    dahl_matrix* sub_matrices;
    bool is_partitioned;
} dahl_block;

// Wrapper around the dahl data structures `dahl_block`, `dahl_matrix` and `dahl_vector`
// that represent any of the previously cited types.
// It can contain a pointer to the original data structure yet it's memory should still
// be managed by the original type.
typedef struct
{
    union dahl_structure
    {
        dahl_block* block;
        dahl_matrix* matrix;
        dahl_vector* vector;
    } structure;

    enum dahl_type
    {
        dahl_type_block,
        dahl_type_matrix,
        dahl_type_vector,
    } type;
} dahl_any;


// Initialize a stack allocated `dahl_any` object from a `dahl_block*`, `dahl_matrix*` or `dahl_vector*`.
#define AS_ANY(x) _Generic((x),                               \
        dahl_block*:                                          \
            (dahl_any)                                        \
            {                                                 \
                .structure = { .block = (dahl_block*)(x) },   \
                .type = dahl_type_block                       \
            },                                                \
        dahl_matrix*:                                         \
            (dahl_any)                                        \
            {                                                 \
                .structure = { .matrix = (dahl_matrix*)(x) }, \
                .type = dahl_type_matrix                      \
            },                                                \
        dahl_vector*:                                         \
            (dahl_any)                                        \
            {                                                 \
                .structure = { .vector = (dahl_vector*)(x) }, \
                .type = dahl_type_vector                      \
            }                                                 \
    )

#define AS_BLOCK(x) (dahl_block*)((x).structure.block)
#define AS_MATRIX(x) (dahl_matrix*)((x).structure.matrix)
#define AS_VECTOR(x) (dahl_vector*)((x).structure.vector)

// Initialize a dahl_block with every values at 0.
// parameters:
// - shape: shape3d object describing the dimensions of the block
dahl_block* block_init(shape3d const shape);

// Initialize a dahl_block with random values.
// parameters:
// - shape: shape3d object describing the dimensions of the block
dahl_block* block_init_random(shape3d const shape);

// Initialize a dahl_block by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: shape3d object describing the dimensions of the block
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_block* block_init_from(shape3d const shape, dahl_fp* const data);

// Returns the block shape
shape3d block_get_shape(dahl_block const* const block);

// Compares the two blocks value by value and returns wether or not they're equal.
bool block_equals(dahl_block const* const a, dahl_block const* const b);

// Partition data along z axis, the sub matrices can then be accesed with `block_get_sub_matrix`.
// Exactly creates z sub matrices, so `block_get_sub_matrix_nb` should be equal to z.
// Note the the block itself cannot be used as long as it is partitioned. TODO: I think?
void block_partition_along_z(dahl_block* const block);

// Unpartition a block
void block_unpartition(dahl_block* const block);

// Get the number of sub matrices
size_t block_get_sub_matrix_nb(dahl_block const* const block);

// Get sub matrix at index
dahl_matrix* block_get_sub_matrix(dahl_block const* const block, const size_t index);

void block_print(dahl_block const* const);
void block_finalize(dahl_block* block);

// Initialize a dahl_matrix with every values at 0.
// parameters:
// - shape: shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init(shape2d const shape);

// Initialize a dahl_matrix with random values.
// parameters:
// - shape: shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init_random(shape2d const shape);

// Initialize a dahl_matrix by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: shape2d object describing the dimensions of the matrix
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_matrix* matrix_init_from(shape2d const shape, dahl_fp* const data);

// Returns the matrix shape
shape2d matrix_get_shape(dahl_matrix const *const matrix);

// Compares the two matrices value by value and returns wether or not they're equal.
bool matrix_equals(dahl_matrix const* const matrix_a, dahl_matrix const* const matrix_b);

void matrix_partition_along_y(dahl_matrix* const matrix);
void matrix_unpartition(dahl_matrix* const matrix);
size_t matrix_get_sub_vector_nb(dahl_matrix const* const matrix);
dahl_vector* matrix_get_sub_matrix(dahl_matrix const* const matrix, const size_t index);

void matrix_print(dahl_matrix const* const matrix);
void matrix_finalize(dahl_matrix* matrix);


// Initialize a dahl_vector with every values at 0.
// parameters:
// - len: size_t lenght of the vector
dahl_vector* vector_init(size_t const len);

// Initialize a dahl_vector with random values.
// parameters:
// - shape: shape2d object describing the dimensions of the vector
dahl_vector* vector_init_random(size_t const len);

// Initialize a dahl_vector by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: shape2d object describing the dimensions of the vector
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_vector* vector_init_from(size_t const len, dahl_fp* const data);

// Returns the vector shape
shape2d vector_get_shape(dahl_vector const *const vector);

// Compares the two matrices value by value and returns wether or not they're equal.
bool vector_equals(dahl_vector const* const vector_a, dahl_vector const* const vector_b);

void vector_print(dahl_vector const* const vector);
void vector_finalize(dahl_vector* vector);

starpu_data_handle_t any_get_handle(dahl_any const any);

// Allocate and return a `dahl_any` with the same shape as the parameter
dahl_any any_zeros_like(dahl_any const any);

#endif //!DAHL_TYPES_H
