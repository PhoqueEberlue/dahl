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

// Different types
typedef enum {
    DAHL_NONE, // Nothing
    DAHL_TENSOR,
    DAHL_BLOCK,
    DAHL_MATRIX,
    DAHL_VECTOR,
} dahl_type;

// Definitions of dahl data structures that were previously defined as opaque types in dahl_data.h
// so their fields are not accessible from the public API.
typedef struct _dahl_vector
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    uint8_t partition_level;
} dahl_vector;

typedef struct _dahl_matrix
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    union {
        dahl_matrix* matrices;
        dahl_vector* vectors;
    } sub_data;

    dahl_type partition_type;

    uint8_t partition_level;
} dahl_matrix;

typedef struct _dahl_block
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    union {
        dahl_block* blocks;
        dahl_matrix* matrices;
        dahl_vector* vectors;
    } sub_data;

    // The type inside the sub_data union.
    dahl_type partition_type;

    // Describes the current object partition level, the father data is always at 0, and the children at 1+.
    // 0 = Not sub data, 1 = sub data of one parent data, 2+ sub data of more than one parent
    uint8_t partition_level;
} dahl_block;

typedef struct _dahl_tensor
{
    starpu_data_handle_t handle;
    dahl_fp* data;

    union {
        dahl_tensor* tensors;
        dahl_block* blocks;
        dahl_matrix* matrices;
        dahl_vector* vectors;
    } sub_data;

    dahl_type partition_type;

    uint8_t partition_level;
} dahl_tensor;


starpu_data_handle_t _vector_get_handle(void const* vector);
starpu_data_handle_t _block_get_handle(void const* block);
starpu_data_handle_t _matrix_get_handle(void const* matrix);
starpu_data_handle_t _tensor_get_handle(void const* tensor);

size_t _tensor_get_nb_elem(void const* tensor);
size_t _block_get_nb_elem(void const* block);
size_t _matrix_get_nb_elem(void const* matrix);
size_t _vector_get_nb_elem(void const* vector);

#endif //!DAHL_DATA_STRUCTURES_H
