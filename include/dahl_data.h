#ifndef DAHL_DATA_H
#define DAHL_DATA_H

#include "dahl_types.h"


typedef struct _dahl_vector dahl_vector;
typedef struct _dahl_matrix dahl_matrix;
typedef struct _dahl_block dahl_block;

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

// Initialize a dahl_block with every values at 0.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init(dahl_shape3d const shape);

// Initialize a dahl_block with random values.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init_random(dahl_shape3d const shape);

// Initialize a dahl_block by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape3d object describing the dimensions of the block
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_block* block_init_from(dahl_shape3d const shape, dahl_fp* const data);

dahl_block* block_clone(dahl_block const* const block);

dahl_block* block_add_padding_init(dahl_block const* const block, dahl_shape3d const new_shape);

// Returns the block shape
dahl_shape3d block_get_shape(dahl_block const* const block);

// Compares the two blocks value by value and returns wether or not they're equal.
bool block_equals(dahl_block const* const a, dahl_block const* const b);

dahl_fp* block_data_acquire(dahl_block const* const block);
void block_data_release(dahl_block const* const block);

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

// Returns a flattened vector of the block, the previous instance of the block is finalized automatically.
dahl_vector* block_to_vector(dahl_block* block);

void block_print(dahl_block const* const);
void block_finalize(dahl_block* block);

// Initialize a dahl_matrix with every values at 0.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init(dahl_shape2d const shape);

// Initialize a dahl_matrix with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init_random(dahl_shape2d const shape);

// Initialize a dahl_matrix by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the matrix
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_matrix* matrix_init_from(dahl_shape2d const shape, dahl_fp* const data);

dahl_matrix* matrix_clone(dahl_matrix const* const matrix);

// Returns the matrix shape
dahl_shape2d matrix_get_shape(dahl_matrix const *const matrix);

// Compares the two matrices value by value and returns wether or not they're equal.
bool matrix_equals(dahl_matrix const* const matrix_a, dahl_matrix const* const matrix_b);

void matrix_partition_along_y(dahl_matrix* const matrix);
void matrix_unpartition(dahl_matrix* const matrix);
size_t matrix_get_sub_vector_nb(dahl_matrix const* const matrix);
dahl_vector* matrix_get_sub_vector(dahl_matrix const* const matrix, const size_t index);

void matrix_print(dahl_matrix const* const matrix);
void matrix_print_ascii(dahl_matrix const* const matrix, dahl_fp const threshold);
void matrix_finalize(dahl_matrix* matrix);


// Initialize a dahl_vector with every values at 0.
// parameters:
// - len: size_t lenght of the vector
dahl_vector* vector_init(size_t const len);

// Initialize a dahl_vector with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the vector
dahl_vector* vector_init_random(size_t const len);

// Initialize a dahl_vector by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the vector
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_vector* vector_init_from(size_t const len, dahl_fp* const data);

dahl_vector* vector_clone(dahl_vector const* const vector);

// Returns the vector len
size_t vector_get_len(dahl_vector const *const vector);

// Converts a vector to a matrix
dahl_matrix* vector_to_matrix(dahl_vector* vector, dahl_shape2d shape);
dahl_matrix* vector_to_column_matrix(dahl_vector* vector);
dahl_matrix* vector_to_row_matrix(dahl_vector* vector);

// Converts a vector to a block
dahl_block* vector_to_block(dahl_vector* vector, dahl_shape3d shape);

// Clone the vector as a categorical matrix
dahl_matrix* vector_as_categorical(dahl_vector* vector, size_t const num_classes);

// Compares the two matrices value by value and returns wether or not they're equal.
// Note: values are rounded in order to obtain valid comparisons.
bool vector_equals(dahl_vector const* const vector_a, dahl_vector const* const vector_b, bool const rounding);

void vector_print(dahl_vector const* const vector);
void vector_finalize_without_data(dahl_vector* vector);
void vector_finalize(dahl_vector* vector);

// -------------------------------------------------- Any operations --------------------------------------------------
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
            },                                                \
        dahl_block const*:                              \
            (dahl_any const)                                  \
            {                                                 \
                .structure = { .block = (dahl_block*)(X) },   \
                .type = dahl_type_block                       \
            },                                                \
        dahl_matrix const*:                             \
            (dahl_any const)                                  \
            {                                                 \
                .structure = { .matrix = (dahl_matrix*)(X) }, \
                .type = dahl_type_matrix                      \
            },                                                \
        dahl_vector const*:                             \
            (dahl_any const)                                  \
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
            },                           \
        dahl_block const*:               \
            (dahl_block*)                \
            {                            \
                (OUT).structure.block    \
            },                           \
        dahl_matrix const*:              \
            (dahl_matrix*)               \
            {                            \
                (OUT).structure.matrix   \
            },                           \
        dahl_vector const*:              \
            (dahl_vector*)               \
            {                            \
                (OUT).structure.vector   \
            }                            \
    )   // TODO: is `default` required?

dahl_fp* any_data_acquire(dahl_any const any);
#define ANY_DATA_ACQUIRE(X) any_data_acquire(AS_ANY(X))

void any_data_release(dahl_any const any);
#define ANY_DATA_RELEASE(X) any_data_release(AS_ANY(X))

dahl_any any_clone(dahl_any const any);
#define ANY_CLONE(X) FROM_ANY(X, any_clone(AS_ANY(X)))

#endif //!DAHL_DATA_H
