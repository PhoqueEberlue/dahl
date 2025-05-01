#include "../../include/dahl_data.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

dahl_vector* vector_init(dahl_arena* arena, size_t const len)
{
    // Arena returns 0 initialized data, no need to declare the elements of the vector
    dahl_fp* data = arena_put(arena, len * sizeof(dahl_fp));
    dahl_vector* vector = arena_put(arena, sizeof(dahl_vector));

    vector->data = data;
    vector->len = len;
    vector->is_sub_matrix_data = false;

    return vector;
}

dahl_vector* vector_init_from(dahl_arena* arena, size_t const len, dahl_fp const* data)
{
    dahl_vector* vector = vector_init(arena, len);
    memcpy(vector->data, data, len);
    return vector;
}

dahl_vector* vector_init_random(dahl_arena* arena, size_t const len)
{
    dahl_vector* vector = vector_init(arena, len);

    for (int i = 0; i < len; i += 1)
    {
        vector->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return vector;
}

dahl_vector* vector_clone(dahl_arena* arena, dahl_vector const* vector)
{
    return vector_init_from(arena, vector->len, vector->data);
}

bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding)
{
    assert(a->len == b->len);
    bool res = true;

    for (int i = 0; i < a->len; i++)
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

void vector_print(dahl_vector const* vector)
{
    printf("vector=%p nx=%zu\n", vector->data, vector->len);

    for(size_t x = 0; x < vector->len; x++)
    {
        printf("%e ", vector->data[x]);
    }
    printf("\n");
}

// dahl_matrix* vector_to_matrix(dahl_vector* vector, dahl_shape2d shape)
// {
//     size_t len = vector_get_len(vector);
//     dahl_fp* data = vector_data_acquire(vector);
// 
//     assert(shape.x * shape.y == len);
// 
//     dahl_matrix* res = matrix_init_from_ptr(shape, data);
// 
//     vector_data_release(vector);
//     vector_finalize_without_data(vector);
// 
//     return res;
// }
// 
// dahl_matrix* vector_to_column_matrix(dahl_vector* vector)
// {
//     size_t len = vector_get_len(vector);
//     dahl_shape2d new_shape = { .x = 1, .y = len };
//     return vector_to_matrix(vector, new_shape);
// }
// 
// dahl_matrix* vector_to_row_matrix(dahl_vector* vector)
// {
//     size_t len = vector_get_len(vector);
//     dahl_shape2d new_shape = { .x = len, .y = 1 };
//     return vector_to_matrix(vector, new_shape);
// }
// 
// dahl_block* vector_to_block(dahl_vector* vector, dahl_shape3d shape)
// {
//     size_t len = vector_get_len(vector);
//     dahl_fp* data = vector_data_acquire(vector);
// 
//     assert(shape.x * shape.y * shape.z == len);
// 
//     dahl_block* res = block_init_from_ptr(shape, data);
// 
//     vector_data_release(vector);
//     vector_finalize_without_data(vector);
// 
//     return res;
// }
// 
// // TODO: why wouldn't it be a codelet?
// dahl_matrix* vector_as_categorical(dahl_vector const* vector, size_t const num_classes)
// {
//     dahl_fp* vec = vector_data_acquire(vector);
//     size_t len = vector_get_len(vector);
// 
//     dahl_shape2d shape = { .x = num_classes, .y = len };
//     dahl_matrix* res = matrix_init(shape);
// 
//     dahl_fp* mat = matrix_data_acquire(res);
// 
//     for (size_t i = 0; i < len; i++)
//     {
//         size_t class = (size_t)vec[i];
//         mat[(i * num_classes) + class] = 1;
//     }
// 
//     vector_data_release(vector);
//     matrix_data_release(res);
// 
//     return res;
// }
