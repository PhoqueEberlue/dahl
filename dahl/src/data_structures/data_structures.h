#ifndef DAHL_DATA_STRUCTURES_H
#define DAHL_DATA_STRUCTURES_H

#include "../../include/dahl_data.h"

#include "../arena/arena.h"
#include "starpu_data.h"
#include "sys/types.h"
#include <starpu.h>
#include <stdint.h>

// In case we want to fix the value for the tests
#ifdef DAHL_TESTS_H
#define DAHL_MAX_RANDOM_VALUES 1
#else
#define DAHL_MAX_RANDOM_VALUES 1
#endif

// Partitionning data
typedef struct _dahl_partition
{
    dahl_traits* trait;
    dahl_access access;
    dahl_partition_type type;

    // if the partition is active
    bool is_active;

    // Handle of the parent data structure
    starpu_data_handle_t main_handle;

    // Handles for each children
    starpu_data_handle_t* handles;

    size_t nb_children;
    // Children list, can be any dahl data structure
    void* children[];
} dahl_partition;

typedef const struct _dahl_traits
{
    void* (*init_from_ptr)(dahl_arena*, starpu_data_handle_t, dahl_fp*);
    starpu_data_handle_t (*get_handle)(void const*);
    size_t (*get_nb_elem)(void const*);
    void (*print_file)(void const*, FILE*, int8_t const);
    bool (*get_is_redux)(void const*);
    void (*enable_redux)(void*);
    dahl_partition* (*get_partition)(void const*);
    dahl_type type;
} dahl_traits;

// Definitions of dahl data structures that were previously defined as opaque types in dahl_data.h
// so their fields are not accessible from the public API.
typedef struct _dahl_scalar
{
    starpu_data_handle_t handle;
    dahl_fp data;
    // Where the data was originally allocated
    dahl_arena* origin_arena;
    bool is_redux;
} dahl_scalar;

typedef struct _dahl_vector
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    // Where the data was originally allocated
    dahl_arena* origin_arena;
    bool is_redux;
} dahl_vector;

typedef struct _dahl_matrix
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    // Where the data was originally allocated
    dahl_arena* origin_arena;
    dahl_partition** partition;
    bool is_redux;
} dahl_matrix;

typedef struct _dahl_block
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    // Where the data was originally allocated
    dahl_arena* origin_arena;
    dahl_partition** partition;
    bool is_redux;
} dahl_block;

typedef struct _dahl_tensor
{
    starpu_data_handle_t handle;
    dahl_fp* data;
    // Where the data was originally allocated
    dahl_arena* origin_arena;
    dahl_partition** partition;
    bool is_redux;
} dahl_tensor;

void* _tensor_init_from_ptr(dahl_arena*, starpu_data_handle_t, dahl_fp* data);
void* _block_init_from_ptr(dahl_arena*, starpu_data_handle_t, dahl_fp* data);
void* _matrix_init_from_ptr(dahl_arena*, starpu_data_handle_t, dahl_fp* data);
void* _vector_init_from_ptr(dahl_arena*, starpu_data_handle_t, dahl_fp* data);

starpu_data_handle_t _tensor_data_register(dahl_arena*, dahl_shape4d shape, dahl_fp* data);
starpu_data_handle_t _block_data_register(dahl_arena*, dahl_shape3d shape, dahl_fp* data);
starpu_data_handle_t _matrix_data_register(dahl_arena*, dahl_shape2d shape, dahl_fp* data);
starpu_data_handle_t _vector_data_register(dahl_arena*, size_t len, dahl_fp* data);

starpu_data_handle_t _tensor_get_handle(void const* tensor);
starpu_data_handle_t _block_get_handle(void const* block);
starpu_data_handle_t _matrix_get_handle(void const* matrix);
starpu_data_handle_t _vector_get_handle(void const* vector);
starpu_data_handle_t _scalar_get_handle(void const* scalar);

size_t _tensor_get_nb_elem(void const* tensor);
size_t _block_get_nb_elem(void const* block);
size_t _matrix_get_nb_elem(void const* matrix);
size_t _vector_get_nb_elem(void const* vector);
size_t _scalar_get_nb_elem(__attribute__((unused))void const* scalar); // Defined just for compatibility

bool _tensor_get_is_redux(void const* tensor);
bool _block_get_is_redux(void const* block);
bool _matrix_get_is_redux(void const* matrix);
bool _vector_get_is_redux(void const* vector);
bool _scalar_get_is_redux(void const* scalar);

void _tensor_enable_redux(void* tensor);
void _block_enable_redux(void* block);
void _matrix_enable_redux(void* matrix);
void _vector_enable_redux(void* vector);
void _scalar_enable_redux(void* scalar);

void _tensor_print_file(void const*, FILE*, int8_t const precision);
void _block_print_file(void const*, FILE*, int8_t const precision);
void _matrix_print_file(void const*, FILE*, int8_t const precision);
void _vector_print_file(void const*, FILE*, int8_t const precision);
void _scalar_print_file(void const*, FILE*, int8_t const precision);

// ---------------------------- Partition related ----------------------------
dahl_partition* _tensor_get_partition(void const* tensor);
dahl_partition* _block_get_partition(void const* block);
dahl_partition* _matrix_get_partition(void const* matrix);

// Init a new partition object in the same arena as the parent
dahl_partition* _partition_init(size_t nb_children, dahl_access access, dahl_traits* trait,
                                struct starpu_data_filter* f, starpu_data_handle_t main_handle,
                                dahl_arena* origin_arena, dahl_partition_type type);

void _partition_submit(dahl_partition* p);
void _unpartition_submit(dahl_partition* p);

#endif //!DAHL_DATA_STRUCTURES_H
