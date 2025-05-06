#ifndef DAHL_DATA_H
#define DAHL_DATA_H

#ifdef DAHL_TESTS_H
#define DAHL_MAX_RANDOM_VALUES 10
#else
#define DAHL_MAX_RANDOM_VALUES 4
#endif

#include "dahl_types.h"
#include "dahl_arena.h"

typedef struct
{
    dahl_fp* data;
    size_t len;

    // Wether this vector is a sub matrix data
    bool is_sub_matrix_data;
} dahl_vector;

typedef struct
{
    dahl_fp* data;
    dahl_shape2d shape;
    size_t ld;

    dahl_vector* sub_vectors;
    size_t nb_sub_vectors;
    bool is_partitioned;

    // Wether this matrix is a sub block data
    bool is_sub_block_data;
} dahl_matrix;

typedef struct
{
    dahl_fp* data;
    dahl_shape3d shape;
    size_t ldz;
    size_t ldy;

    dahl_matrix* sub_matrices;
    size_t nb_sub_matrices;
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

// Initialize a dahl_block with every values at 0.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init(dahl_arena* arena, dahl_shape3d const shape);

// Initialize a dahl_block with random values.
// parameters:
// - shape: dahl_shape3d object describing the dimensions of the block
dahl_block* block_init_random(dahl_arena* arena, dahl_shape3d const shape);

// Initialize a dahl_block by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape3d object describing the dimensions of the block
// - data: pointer to contiguous allocated dahl_fp array with x*y*z number of elements
dahl_block* block_init_from(dahl_arena* arena, dahl_shape3d const shape, dahl_fp const* data);

dahl_block* block_clone(dahl_arena* arena, dahl_block const* block);

// Returns a new block with added padding. 
// The new_shape should be larger than the previous block.
// If it is exactly the same, it just produces a copy of the bolck.
// If the new padding is even, the remainder is placed at the end of the axis.
dahl_block* block_add_padding_init(dahl_arena* arena, dahl_block const* block, dahl_shape3d const new_shape);

// Compares the two blocks value by value and returns wether or not they're equal.
bool block_equals(dahl_block const* a, dahl_block const* b, bool const rounding);

// dahl_fp* block_data_acquire(dahl_block const* block);
// void block_data_release(dahl_block const* block);

// Partition data along z axis, the sub matrices can then be accesed with `block_get_sub_matrix`.
// Exactly creates z sub matrices, so `block_get_sub_matrix_nb` should be equal to z.
// Note the the block itself cannot be used as long as it is partitioned. TODO: I think?
void block_partition_along_z(dahl_arena* arena, dahl_block* block);

// Unpartition a block
void block_unpartition(dahl_block* block);

//TODO this may be deleted
// Get sub matrix at index
dahl_matrix* block_get_sub_matrix(dahl_block const* block, const size_t index);

// // Returns a flattened vector of the block, the previous instance of the block is finalized automatically.
dahl_vector* block_to_vector(dahl_arena* arena, dahl_block* block);

void block_print(dahl_block const* block);

// Initialize a dahl_matrix with every values at 0.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init(dahl_arena* arena, dahl_shape2d const shape);

// Initialize a dahl_matrix with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the matrix
dahl_matrix* matrix_init_random(dahl_arena* arena, dahl_shape2d const shape);

// Initialize a dahl_matrix by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the matrix
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_matrix* matrix_init_from(dahl_arena* arena, dahl_shape2d const shape, dahl_fp const* data);

dahl_matrix* matrix_clone(dahl_arena* arena, dahl_matrix const* matrix);

// Compares the two matrices value by value and returns wether or not they're equal.
bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding);

void matrix_partition_along_y(dahl_matrix* matrix);
void matrix_unpartition(dahl_matrix* matrix);
dahl_vector* matrix_get_sub_vector(dahl_matrix const* matrix, size_t const index);

void matrix_print(dahl_matrix const* matrix);
void matrix_print_ascii(dahl_matrix const* matrix, dahl_fp const threshold);


// Initialize a dahl_vector with every values at 0.
// parameters:
// - len: size_t lenght of the vector
dahl_vector* vector_init(dahl_arena* arena, size_t const len);

// Initialize a dahl_vector with random values.
// parameters:
// - shape: dahl_shape2d object describing the dimensions of the vector
dahl_vector* vector_init_random(dahl_arena* arena, size_t const len);

// Initialize a dahl_vector by cloning an existing array.
// Cloned memory will be freed upon calling `block_finalize`, however do not forget to free the original array.
// - shape: dahl_shape2d object describing the dimensions of the vector
// - data: pointer to contiguous allocated dahl_fp array with x*y number of elements
dahl_vector* vector_init_from(dahl_arena* arena, size_t const len, dahl_fp const* data);

dahl_vector* vector_clone(dahl_arena* arena, dahl_vector const* vector);

// TODO
// // Converts a vector to a matrix
// dahl_matrix* vector_to_matrix(dahl_vector* vector, dahl_shape2d shape);
// dahl_matrix* vector_to_column_matrix(dahl_vector* vector);
// dahl_matrix* vector_to_row_matrix(dahl_vector* vector);
// 
// // Converts a vector to a block
// dahl_block* vector_to_block(dahl_vector* vector, dahl_shape3d shape);
// 
// // Clone the vector as a categorical matrix
// dahl_matrix* vector_as_categorical(dahl_vector const* vector, size_t const num_classes);

// Compares the two matrices value by value and returns wether or not they're equal.
// Note: values are rounded in order to obtain valid comparisons.
bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding);

void vector_print(dahl_vector const* vector);

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

#endif //!DAHL_DATA_H
