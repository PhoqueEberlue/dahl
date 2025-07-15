#ifndef DAHL_DATA_STRUCTURES_H
#define DAHL_DATA_STRUCTURES_H

#include "../../include/dahl_data.h"

#include "../arena/arena.h"
#include "starpu_data.h"
#include <starpu.h>
#include <stdint.h>

// In case we want to fix the value for the tests
#ifdef DAHL_TESTS_H
#define DAHL_MAX_RANDOM_VALUES 1
#else
#define DAHL_MAX_RANDOM_VALUES 1
#endif

typedef const struct _dahl_traits
{
    void* (*init_from_ptr)(starpu_data_handle_t handle, dahl_fp* data);
    starpu_data_handle_t (*get_handle)(void const*);
    dahl_partition* (*get_partition)(void const*);
    size_t (*get_nb_elem)(void const*);
    dahl_type type;
} dahl_traits;

// Partitionning data
typedef struct _dahl_partition
{
    dahl_type type;
    bool is_mut;
    starpu_data_handle_t* handles;
    size_t nb_children;

    void* children[];
} dahl_partition;

typedef struct
{
    // Where the corresponding data was allocated
    dahl_arena* origin_arena;
    int8_t current_partition;
    dahl_partition* partitions[];
} metadata;

#define TENSOR_NB_PARTITION_TYPE 1
#define BLOCK_NB_PARTITION_TYPE 5
#define MATRIX_NB_PARTITION_TYPE 2

typedef enum 
{
    TENSOR_PARTITION_ALONG_T,
} tensor_partition_type;

typedef enum 
{
    BLOCK_PARTITION_ALONG_Z,
    BLOCK_PARTITION_ALONG_Z_FLAT_MATRICES,
    BLOCK_PARTITION_ALONG_Z_FLAT_VECTORS,
    BLOCK_PARTITION_ALONG_Z_BATCH,
    BLOCK_PARTITION_FLATTEN_TO_VECTOR,
} block_partition_type;

typedef enum 
{
    MATRIX_PARTITION_ALONG_Y,
    MATRIX_PARTITION_ALONG_Y_BATCH,
} matrix_partition_type;

// Definitions of dahl data structures that were previously defined as opaque types in dahl_data.h
// so their fields are not accessible from the public API.
typedef struct _dahl_vector
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    metadata* meta;
} dahl_vector;

typedef struct _dahl_matrix
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    metadata* meta;
} dahl_matrix;

typedef struct _dahl_block
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    metadata* meta;
} dahl_block;

typedef struct _dahl_tensor
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    metadata* meta;
} dahl_tensor;

dahl_partition* _partition_init(size_t nb_children, bool is_mut, dahl_traits* trait, 
                                struct starpu_data_filter* f, starpu_data_handle_t main_handle,
                                dahl_arena* origin_arena);

void _partition_submit_if_needed(metadata* meta, int8_t index, bool should_be_mut, starpu_data_handle_t main_handle);

void* _tensor_init_from_ptr(starpu_data_handle_t handle, dahl_fp* data);
void* _block_init_from_ptr(starpu_data_handle_t handle, dahl_fp* data);
void* _matrix_init_from_ptr(starpu_data_handle_t handle, dahl_fp* data);
void* _vector_init_from_ptr(starpu_data_handle_t handle, dahl_fp* data);


starpu_data_handle_t _vector_get_handle(void const* vector);
starpu_data_handle_t _block_get_handle(void const* block);
starpu_data_handle_t _matrix_get_handle(void const* matrix);
starpu_data_handle_t _tensor_get_handle(void const* tensor);

dahl_partition* _block_get_current_partition(void const* block);
dahl_partition* _matrix_get_current_partition(void const* matrix);
dahl_partition* _tensor_get_current_partition(void const* tensor);

size_t _tensor_get_nb_elem(void const* tensor);
size_t _block_get_nb_elem(void const* block);
size_t _matrix_get_nb_elem(void const* matrix);
size_t _vector_get_nb_elem(void const* vector);

#endif //!DAHL_DATA_STRUCTURES_H
