#ifndef DAHL_DATA_STRUCTURES_H
#define DAHL_DATA_STRUCTURES_H

#include "../../include/dahl_data.h"

#include "../arena/arena.h"
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

    // Wether this vector is matrix (or block) sub data 
    bool is_sub_data;
} dahl_vector;

typedef struct _dahl_matrix
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    dahl_vector* sub_vectors;
    bool is_partitioned;

    // Wether this matrix is a block sub data
    bool is_sub_data;
} dahl_matrix;

typedef struct _dahl_block
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    dahl_matrix* sub_matrices;
    dahl_vector* sub_vectors;
    bool is_partitioned;
} dahl_block;

#endif //!DAHL_DATA_STRUCTURES_H
