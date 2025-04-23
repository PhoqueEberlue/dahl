#include "../include/dahl_data.h"

#include <starpu.h>

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

// Private functions
starpu_data_handle_t vector_get_handle(dahl_vector const* const vector);
starpu_data_handle_t matrix_get_handle(dahl_matrix const* const matrix);
starpu_data_handle_t block_get_handle(dahl_block const* const block);
starpu_data_handle_t any_get_handle(dahl_any const any);

dahl_fp* any_get_data(dahl_any const any);
#define ANY_GET_DATA(X) any_get_data(AS_ANY(X))

// Finalize the block without freeing the pointed data
void block_finalize_without_data(dahl_block* block);
