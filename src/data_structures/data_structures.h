#ifndef DAHL_DATA_STRUCTURES_H
#define DAHL_DATA_STRUCTURES_H

#include "../../include/dahl_data.h"

#include <starpu.h>

// In case we want to fix the value for the tests
#ifdef DAHL_TESTS_H
#define DAHL_MAX_RANDOM_VALUES 1
#else
#define DAHL_MAX_RANDOM_VALUES 1
#endif

// Definitions of dahl data structures that were previously defined as opaque types in dahl_data.h
// so their fields are not accessible from the public API.
typedef struct _dahl_vector
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    // Wether this vector is a sub matrix data
    bool is_sub_matrix_data;
} dahl_vector;

typedef struct _dahl_matrix
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    dahl_vector* sub_vectors;
    bool is_partitioned;

    // Wether this matrix is a sub block data
    bool is_sub_block_data;
} dahl_matrix;

typedef struct _dahl_block
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    dahl_matrix* sub_matrices;
    bool is_partitioned;
} dahl_block;

starpu_data_handle_t any_get_handle(dahl_any const any);

// Those 3 functions are private because we don't want the user to instantiate our data structures
// with data allocated from the outside.
dahl_block* block_init_from_ptr(dahl_shape3d const shape, dahl_fp* data);
dahl_matrix* matrix_init_from_ptr(dahl_shape2d const shape, dahl_fp* data);
dahl_vector* vector_init_from_ptr(size_t const len, dahl_fp* data);

#endif //!DAHL_DATA_STRUCTURES_H
