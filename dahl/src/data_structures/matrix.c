#include "../tasks/codelets.h"
#include "data_structures.h"
#include "custom_filters.h"
#include "starpu_data.h"
#include "starpu_data_interfaces.h"
#include "sys/types.h"
#include <stdio.h>

void* _matrix_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    dahl_matrix* matrix = dahl_arena_alloc(arena, sizeof(dahl_matrix));
    matrix->handle = handle;
    matrix->data = data;
    matrix->origin_arena = arena;
    matrix->is_redux = false;
    matrix->partition = (dahl_partition**)dahl_arena_alloc(arena, sizeof(dahl_partition**));

    return matrix;
}

dahl_fp* _matrix_data_alloc(dahl_arena* arena, dahl_shape2d shape)
{
    size_t n_elems = shape.x * shape.y;
    return dahl_arena_alloc(arena, n_elems * sizeof(dahl_fp));
}

starpu_data_handle_t _matrix_data_register(
        dahl_arena* arena, dahl_shape2d const shape, dahl_fp* data)
{
    starpu_data_handle_t handle = nullptr;
    starpu_matrix_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x,
        shape.y,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(arena, handle);
    return handle;
}

dahl_matrix* matrix_init(dahl_arena* arena, dahl_shape2d const shape)
{
    dahl_fp* data = _matrix_data_alloc(arena, shape);
    memset(data, 0, shape.x*shape.y * sizeof(dahl_fp));
    starpu_data_handle_t handle = _matrix_data_register(arena, shape, data);
    dahl_matrix* matrix = _matrix_init_from_ptr(arena, handle, data);
    return matrix;
}

dahl_matrix* matrix_init_redux(dahl_arena* arena, dahl_shape2d const shape)
{
    dahl_matrix* matrix = matrix_init(arena, shape);
    _matrix_enable_redux(matrix);
    return matrix;
}

dahl_matrix* matrix_init_from(dahl_arena* arena, dahl_shape2d const shape, dahl_fp const* data)
{
    dahl_fp* matrix_data = _matrix_data_alloc(arena, shape);

    for (size_t i = 0; i < shape.x*shape.y; i++)
         matrix_data[i] = data[i];

    starpu_data_handle_t handle = _matrix_data_register(arena, shape, matrix_data);
    return _matrix_init_from_ptr(arena, handle, matrix_data);
}

dahl_matrix* matrix_init_random(
        dahl_arena* arena, dahl_shape2d const shape, dahl_fp min, dahl_fp max)
{
    dahl_fp* matrix_data = _matrix_data_alloc(arena, shape);

    for (size_t y = 0; y < shape.y; y++)
    {
        for (size_t x = 0; x < shape.x; x++)
        {
            matrix_data[(y * shape.x) + x] = fp_rand(min, max);
        }
    }

    starpu_data_handle_t handle = _matrix_data_register(arena, shape, matrix_data);
    return _matrix_init_from_ptr(arena, handle, matrix_data);
}

dahl_tensor* matrix_to_tensor_no_copy(dahl_matrix const* matrix, dahl_shape4d const new_shape)
{
    dahl_shape2d shape = matrix_get_shape(matrix);
    assert(shape.x * shape.y == new_shape.x * new_shape.y * new_shape.z * new_shape.t);

    // Registers our matrix data as a tensor (with new shape), handle will be attached to the
    // matrix's origin arena.
    starpu_data_handle_t handle = _tensor_data_register(
            matrix->origin_arena, new_shape, matrix->data);

    dahl_tensor* res = _tensor_init_from_ptr(matrix->origin_arena, handle, matrix->data);

    // Here we use the same trick when doing manual partitioning:
    // Use cl_switch to force data refresh in our new handle from the tensor handle
	int ret = starpu_task_insert(&cl_switch, STARPU_RW, matrix->handle, STARPU_W, handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "matrix_to_tensor_no_copy");

    // Then deactivate the matrix handle
    starpu_data_invalidate_submit(matrix->handle);

    return res;
}

dahl_tensor_part* matrix_to_tensor_no_copy_partition(
        dahl_matrix* matrix, dahl_shape4d const new_shape)
{
    // Initialize a tensor, just to hold the childrens
    dahl_shape2d shape = matrix_get_shape(matrix);
    assert(shape.x * shape.y == new_shape.x * new_shape.y * new_shape.z * new_shape.t);

    // Registers our matrix data as a tensor (with new shape), handle will be attached to the
    // matrix's origin arena.
    starpu_data_handle_t handle = _tensor_data_register(
            matrix->origin_arena, new_shape, matrix->data);
    
    dahl_tensor* tensor = _tensor_init_from_ptr(matrix->origin_arena, handle, matrix->data);

    size_t const batch_size = GET_NB_CHILDREN(matrix);

    // Create a fake partition to hold our flat children
    dahl_partition* p = dahl_arena_alloc(
        matrix->origin_arena,
        // The partition object itself
        sizeof(dahl_partition) + 
        // + the children array with enough space to store their pointers
        (batch_size * sizeof(void*))
    );

    p->handles = (starpu_data_handle_t*)dahl_arena_alloc(
        matrix->origin_arena,
        batch_size * sizeof(starpu_data_handle_t));
    p->access = DAHL_READ;
    p->nb_children = batch_size;
    p->trait = &dahl_traits_block;

    dahl_shape3d block_shape = { .x = new_shape.x, .y = new_shape.y, .z = new_shape.z };

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_vector const* vector = GET_SUB_VECTOR(matrix, i);
        dahl_block* flat_child = vector_to_block_no_copy(vector, block_shape);
        p->children[i] = flat_child;
        p->handles[i] = flat_child->handle;

        // Invalidate each children of the origin matrix
        starpu_data_invalidate_submit(vector->handle);
    }

    // Pretend the partition is active
    *tensor->partition = p;
    p->is_active = true;
    return tensor;
}

void _matrix_enable_redux(void* matrix)
{
    ((dahl_matrix*)matrix)->is_redux = true;
    starpu_data_set_reduction_methods(
            ((dahl_matrix*)matrix)->handle, &cl_matrix_accumulate, &cl_matrix_zero);
}

bool _matrix_get_is_redux(void const* matrix)
{
    return ((dahl_matrix const*)matrix)->is_redux;
}

void matrix_set_from(dahl_matrix* matrix, dahl_fp const* data)
{
    dahl_shape2d shape = matrix_get_shape(matrix);
    matrix_acquire(matrix);

    size_t i = 0;
    for (size_t y = 0; y < shape.y; y++)
    {
        for (size_t x = 0; x < shape.x; x++)
        {
            matrix_set_value(matrix, x, y, data[i]);
            i++;
        }
    }

    matrix_release(matrix);
}

dahl_shape2d matrix_get_shape(dahl_matrix const* matrix)
{
    size_t nx = starpu_matrix_get_nx(matrix->handle);
    size_t ny = starpu_matrix_get_ny(matrix->handle);
    
    dahl_shape2d res = { .x = nx, .y = ny };
    return res;
}

inline dahl_fp matrix_get_value(dahl_matrix const* matrix, size_t x, size_t y)
{
    size_t ld = starpu_matrix_get_local_ld(matrix->handle);
    return matrix->data[(y * ld) + x];
}

inline void matrix_set_value(dahl_matrix* matrix, size_t x, size_t y, dahl_fp value)
{
    size_t ld = starpu_matrix_get_local_ld(matrix->handle);
    matrix->data[(y * ld) + x] = value;
}

starpu_data_handle_t _matrix_get_handle(void const* matrix)
{
    return ((dahl_matrix*)matrix)->handle;
}

size_t _matrix_get_nb_elem(void const* matrix)
{
    dahl_shape2d shape = matrix_get_shape((dahl_matrix*)matrix);
    return shape.x * shape.y;
}

void matrix_acquire(dahl_matrix const* matrix)
{
    starpu_data_acquire(matrix->handle, STARPU_R);
}

void matrix_acquire_mut(dahl_matrix* matrix)
{
    starpu_data_acquire(matrix->handle, STARPU_RW);
}

void matrix_release(dahl_matrix const* matrix)
{
    starpu_data_release(matrix->handle);
}

bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding, int8_t const precision)
{
    dahl_shape2d const shape_a = matrix_get_shape(a);
    dahl_shape2d const shape_b = matrix_get_shape(b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y);

    matrix_acquire(a);
    matrix_acquire(b);

    bool res = true;

    for (size_t y = 0; y < shape_a.y; y++)
    {
        for (size_t x = 0; x < shape_a.x; x++)
        {
            dahl_fp a_val = matrix_get_value(a, x, y);
            dahl_fp b_val = matrix_get_value(b, x, y);

            if (rounding) { res = fp_equals_round(a_val, b_val, precision); }
            else          { res = fp_equals(a_val, b_val);                  }

            if (!res)     { goto RELEASE; }
        }
    }

RELEASE:
    matrix_release(a);
    matrix_release(b);

    return res;
}

void matrix_to_csv(dahl_matrix const* matrix, char const* file_path, char const** colnames)
{
    dahl_shape2d shape = matrix_get_shape(matrix);
    matrix_acquire(matrix);
    FILE* fp = fopen(file_path, "w");
    // Write header
    for (size_t x = 0; x < shape.x; x++)
    {
        fprintf(fp, ",%s", colnames[x]);
    }

    fprintf(fp, "\n");

    for (size_t y = 0; y < shape.y; y++)
    {
        fprintf(fp, "%lu", y);
        for (size_t x = 0; x < shape.x; x++)
        {
            dahl_fp value = matrix_get_value(matrix, x, y);
            fprintf(fp, ",%f", value);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    matrix_release(matrix);
}

dahl_partition* _matrix_get_partition(void const* matrix)
{
    return *((dahl_matrix*)matrix)->partition;
}

void matrix_partition_along_y(dahl_matrix const* matrix, dahl_access access)
{
    size_t const nparts = matrix_get_shape(matrix).y;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_pick_vector_y,
		.nchildren = nparts,
		.get_child_ops = starpu_matrix_filter_pick_vector_child_ops
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_vector,
                                        &f, matrix->handle, matrix->origin_arena,
                                        MATRIX_PARTITION_ALONG_Y);

    _partition_submit(p);
    *matrix->partition = p;
}

void matrix_partition_along_y_batch(dahl_matrix const* matrix, dahl_access access, size_t batch_size)
{
    size_t const nparts = matrix_get_shape(matrix).y / batch_size;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_matrix,
		.nchildren = nparts,
		.get_child_ops = starpu_matrix_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_matrix,
                                        &f, matrix->handle, matrix->origin_arena,
                                        MATRIX_PARTITION_ALONG_Y_BATCH);

    _partition_submit(p);
    *matrix->partition = p;
}

void matrix_unpartition(dahl_matrix_part const* matrix)
{
    dahl_partition* p = *matrix->partition;
    assert(p && p->is_active);
    _unpartition_submit(p);
}

void _matrix_print_file(void const* vmatrix, FILE* fp, int8_t const precision)
{
    auto matrix = (dahl_matrix const*)vmatrix;
    const dahl_shape2d shape = matrix_get_shape(matrix);

	size_t ld = starpu_matrix_get_local_ld(matrix->handle);

	matrix_acquire(matrix);

    fprintf(fp, "matrix=%p nx=%zu ny=%zu ld=%zu\n{ ", matrix->data, shape.x, shape.y, ld);
    for(size_t y = 0; y < shape.y; y++)
    {
        fprintf(fp, "\n\t{ ");
        for(size_t x = 0; x < shape.x; x++)
        {
            fprintf(fp, "%+.*f, ", precision, matrix_get_value(matrix, x, y));
        }
        fprintf(fp, "},");
    }
    fprintf(fp, "\n}\n");

	matrix_release(matrix);
}

void matrix_print(dahl_matrix const* matrix)
{
    _matrix_print_file(matrix, stdout, DAHL_DEFAULT_PRINT_PRECISION);
}

void matrix_print_ascii(dahl_matrix const* matrix, dahl_fp const threshold)
{
    const dahl_shape2d shape = matrix_get_shape(matrix);
	size_t ld = starpu_matrix_get_local_ld(matrix->handle);
	matrix_acquire(matrix);

    printf("matrix=%p nx=%zu ny=%zu ld=%zu\n", matrix->data, shape.x, shape.y, ld);

    for(size_t y = 0; y < shape.y; y++)
    {
        for(size_t x = 0; x < shape.x; x++)
        {
            dahl_fp value = matrix_get_value(matrix, x, y);

            value < threshold ? printf(". ") : printf("# ");
        }
        printf("\n");
    }
    printf("\n");

	matrix_release(matrix);
}

void matrix_image_display(dahl_matrix const* matrix, size_t const scale_factor)
{
    dahl_shape2d shape = matrix_get_shape(matrix);
    matrix_acquire(matrix);

    char cmd[100] = {};
    // Create a command that uses ImageMagick to display our matrix into an image
    sprintf(cmd, "display -resize %lux%lu -", shape.x * scale_factor, shape.y * scale_factor);

    FILE *fp = popen(cmd, "w");
    // Use PGM format with P5 for grayscale
    fprintf(fp, "P5\n%d %d\n255\n", shape.x, shape.y);

    // normalize to 0..255
    dahl_fp min = matrix_get_value(matrix, 0, 0);
    dahl_fp max = min;
    for (size_t y = 0; y < shape.y; y++)
    {
        for (size_t x = 0; x < shape.x; x++)
        {
            dahl_fp value = matrix_get_value(matrix, x, y);
            if (value < min) { min = value; }
            if (value > max) { max = value; }
        }
    }
    dahl_fp range = max - min;

    for (size_t y = 0; y < shape.y; y++)
    {
        for (size_t x = 0; x < shape.x; x++)
        {
            unsigned char val = (unsigned char)(255.0F * (matrix_get_value(matrix, x, y) - min) / (range + 1e-8F));
            fwrite(&val, 1, 1, fp);
        }
    }

    pclose(fp);
    matrix_release(matrix);
}
