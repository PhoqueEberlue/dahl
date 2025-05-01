#include "../../include/dahl_data.h"
#include "string.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

dahl_matrix* matrix_init(dahl_arena* arena, dahl_shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;

    // Arena returns 0 initialized data, no need to declare the elements of the matrix
    dahl_fp* data = arena_put(arena, n_elems * sizeof(dahl_fp));
    dahl_matrix* matrix = arena_put(arena, sizeof(dahl_matrix));

    matrix->data = data;
    matrix->shape = shape;
    matrix->ld = shape.x;
    matrix->is_sub_block_data = false;
    matrix->is_partitioned = false;
    matrix->sub_vectors = nullptr;

    return matrix;
}

dahl_matrix* matrix_init_from(dahl_arena* arena, dahl_shape2d const shape, dahl_fp const* data)
{
    dahl_matrix* matrix = matrix_init(arena, shape);
    memcpy(matrix->data, data, shape.x * shape.y);
    return matrix;
}

dahl_matrix* matrix_init_random(dahl_arena* arena, dahl_shape2d const shape)
{
    dahl_matrix* matrix = matrix_init(arena, shape);

    for (int i = 0; i < shape.x * shape.y; i += 1)
    {
        matrix->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return matrix;
}

dahl_matrix* matrix_clone(dahl_arena* arena, dahl_matrix const* matrix)
{
    return matrix_init_from(arena, matrix->shape, matrix->data);
}

bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding)
{
    assert(a->shape.x == b->shape.x 
        && a->shape.y == b->shape.y);

    bool res = true;

    for (int i = 0; i < (a->shape.x * a->shape.y); i++)
    {
        if (rounding)
        {
            if (round(a->data[i]) != round(b->data[i]))
            {
                res = false;
                break;
            }
        }
        else 
        {
            if (a->data[i] != b->data[i])
            {
                res = false;
                break;
            }
        }
    }

    return res;
}

void matrix_partition_along_z(dahl_arena* arena, dahl_matrix* matrix)
{
    matrix->is_partitioned = true;
    matrix->nb_sub_vectors = matrix->shape.y;
    matrix->sub_vectors = arena_put(arena, matrix->shape.y * sizeof(dahl_vector));

    for (int y = 0; y < matrix->shape.y; y++)
    {
        matrix->sub_vectors[y].data = &matrix->data[(y * matrix->ld)];
        matrix->sub_vectors[y].is_sub_matrix_data = true;
        matrix->sub_vectors[y].len = matrix->shape.x;
    }
}

void matrix_unpartition(dahl_matrix* matrix)
{
    matrix->sub_vectors = nullptr;
    matrix->is_partitioned = false;
}

dahl_vector* matrix_get_sub_vector(dahl_matrix const* matrix, const size_t index)
{
    assert(matrix->is_partitioned 
        && matrix->sub_vectors != nullptr 
        && index < matrix->nb_sub_vectors);

    return &matrix->sub_vectors[index];
}

void matrix_print(dahl_matrix const* matrix)
{
    printf("matrix=%p nx=%zu ny=%zu ld=%zu\n", matrix->data, matrix->shape.x, matrix->shape.y, matrix->ld);

    for(size_t y = 0; y < matrix->shape.y; y++)
    {
        // printf("%s", space_offset(shape.y - y - 1));

        for(size_t x = 0; x < matrix->shape.x; x++)
        {
            printf("%f ", matrix->data[(y * matrix->ld) + x]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_print_ascii(dahl_matrix const* matrix, dahl_fp const threshold)
{
    printf("matrix=%p nx=%zu ny=%zu ld=%zu\n", matrix->data, matrix->shape.x, matrix->shape.y, matrix->ld);

    for(size_t y = 0; y < matrix->shape.y; y++)
    {
        // printf("%s", space_offset(shape.y - y - 1));

        for(size_t x = 0; x < matrix->shape.x; x++)
        {
            dahl_fp value = matrix->data[(y * matrix->ld) + x];

            value < threshold ? printf(". ") : printf("# ");
        }
        printf("\n");
    }
}
