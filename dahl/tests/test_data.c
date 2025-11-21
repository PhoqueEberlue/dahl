#include "tests.h"
#include "unistd.h"
#include <stdio.h>

void test_tensor_partition_along_t()
{
    dahl_shape4d data_shape = { .x = 4, .y = 3, .z = 2, .t = 2 };

    dahl_fp data[2][2][3][4] = {
        {
            {
                {-2, 1, 2,-1 },
                { 3, 1,-3, 1 },
                { 4,-1, 4,-1 },
            },
            {
                { 3, 1,-8,-3 },
                {-7,-3, 3, 2 },
                { 1, 1, 9, 1 },
            },
        },
        {
            {
                { 2,-1,-2, 1 },
                {-3,-1, 3,-1 },
                {-4, 1,-4, 1 },
            },
            {
                {-3,-1, 8, 3 },
                { 7, 3,-3,-2 },
                {-1,-1,-9,-1 },
            },
        }
    };

    dahl_tensor* tensor = tensor_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_shape3d expect_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp expect_0[2][3][4] = {
        {
            {-2, 1, 2,-1 },
            { 3, 1,-3, 1 },
            { 4,-1, 4,-1 },
        },
        {
            { 3, 1,-8,-3 },
            {-7,-3, 3, 2 },
            { 1, 1, 9, 1 },
        },
    };

    dahl_fp expect_1[2][3][4] = {
        {
            { 2,-1,-2, 1 },
            {-3,-1, 3,-1 },
            {-4, 1,-4, 1 },
        },
        {
            {-3,-1, 8, 3 },
            { 7, 3,-3,-2 },
            {-1,-1,-9,-1 },
        },
    };

    dahl_block const* expect_block_0 = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect_0);
    dahl_block const* expect_block_1 = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect_1);

    dahl_tensor_p* tensor_p = tensor_partition_along_t(tensor, DAHL_READ);

    dahl_block const* sub_block_0 = GET_SUB_BLOCK(tensor_p, 0);
    dahl_shape3d shape_0 = block_get_shape(sub_block_0);
    ASSERT_SHAPE3D_EQUALS(expect_shape, shape_0);
    ASSERT_BLOCK_EQUALS(expect_block_0, sub_block_0);

    dahl_block const* sub_block_1 = GET_SUB_BLOCK(tensor_p, 1);
    dahl_shape3d shape_1 = block_get_shape(sub_block_1);
    ASSERT_SHAPE3D_EQUALS(expect_shape, shape_1);
    ASSERT_BLOCK_EQUALS(expect_block_1, sub_block_1);

    tensor_unpartition(tensor_p);

    dahl_arena_reset(testing_arena);
}

void test_tensor_partition_along_t_batch()
{
    dahl_shape4d data_shape = { .x = 2, .y = 1, .z = 1, .t = 4 };

    dahl_fp data[4][1][1][2] = {
        { { { 1, 2 }, }, },
        { { { 3, 4 }, }, },
        { { { 5, 6 }, }, },
        { { { 7, 8 }, }, }
    };

    dahl_tensor* tensor = tensor_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_shape4d expect_shape = { .x = 2, .y = 1, .z = 1, .t = 2 };

    dahl_fp expect_0[2][1][1][2] = { 
        { { { 1, 2 }, }, },
        { { { 3, 4 }, }, }
    };

    dahl_fp expect_1[2][1][1][2] = { 
        { { { 5, 6 }, }, },
        { { { 7, 8 }, }, }
    };

    dahl_tensor const* expect_tensor_0 = tensor_init_from(testing_arena, expect_shape, (dahl_fp*)&expect_0);
    dahl_tensor const* expect_tensor_1 = tensor_init_from(testing_arena, expect_shape, (dahl_fp*)&expect_1);

    size_t const batch_size = 2;
    dahl_tensor_p* tensor_p = tensor_partition_along_t_batch(tensor, DAHL_READ, batch_size);

    dahl_tensor const* sub_tensor_0 = GET_SUB_TENSOR(tensor_p, 0);
    dahl_shape4d shape_0 = tensor_get_shape(sub_tensor_0);
    ASSERT_SHAPE4D_EQUALS(expect_shape, shape_0);
    ASSERT_TENSOR_EQUALS(expect_tensor_0, sub_tensor_0);

    dahl_tensor const* sub_tensor_1 = GET_SUB_TENSOR(tensor_p, 1);
    dahl_shape4d shape_1 = tensor_get_shape(sub_tensor_1);
    ASSERT_SHAPE4D_EQUALS(expect_shape, shape_1);
    ASSERT_TENSOR_EQUALS(expect_tensor_1, sub_tensor_1);

    tensor_unpartition(tensor_p);

    dahl_arena_reset(testing_arena);
}

void test_block_partition_along_z()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2, 1, 2,-1 },
            { 3, 1,-3, 1 },
            { 4,-1, 4,-1 },
        },
        {
            { 3, 1,-8,-3 },
            {-7,-3, 3, 2 },
            { 1, 1, 9, 1 },
        },
    };

    dahl_block* block = block_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_shape2d expect_shape = { .x = 4, .y = 3 };

    dahl_fp expect_0[3][4] = {
        {-2, 1, 2,-1 },
        { 3, 1,-3, 1 },
        { 4,-1, 4,-1 },
    };

    dahl_fp expect_1[3][4] = {
        { 3, 1,-8,-3 },
        {-7,-3, 3, 2 },
        { 1, 1, 9, 1 },
    };

    dahl_matrix const* expect_matrix_0 = matrix_init_from(testing_arena, expect_shape, (dahl_fp*)&expect_0);
    dahl_matrix const* expect_matrix_1 = matrix_init_from(testing_arena, expect_shape, (dahl_fp*)&expect_1);

    dahl_block_p* block_p = block_partition_along_z(block, DAHL_READ);

    dahl_matrix const* sub_matrix_0 = GET_SUB_MATRIX(block_p, 0);
    dahl_shape2d shape_0 = matrix_get_shape(sub_matrix_0);
    ASSERT_SHAPE2D_EQUALS(expect_shape, shape_0);
    ASSERT_MATRIX_EQUALS(expect_matrix_0, sub_matrix_0);

    dahl_matrix const* sub_matrix_1 = GET_SUB_MATRIX(block_p, 1);
    dahl_shape2d shape_1 = matrix_get_shape(sub_matrix_1);
    ASSERT_SHAPE2D_EQUALS(expect_shape, shape_1);
    ASSERT_MATRIX_EQUALS(expect_matrix_1, sub_matrix_1);

    block_unpartition(block_p);

    dahl_arena_reset(testing_arena);
}

void test_block_partition_flatten_to_vector()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2, 1, 2,-1 },
            { 3, 1,-3, 1 },
            { 4,-1, 4,-1 },
        },
        {
            { 3, 1,-8,-3 },
            {-7,-3, 3, 2 },
            { 1, 1, 9, 1 },
        },
    };

    dahl_block* block = block_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_fp expect[24] = {
        -2, 1, 2,-1,
        3, 1,-3, 1,
        4,-1, 4,-1,
        3, 1,-8,-3,
       -7,-3, 3, 2,
        1, 1, 9, 1,
    };

    dahl_vector const* expect_vector = vector_init_from(testing_arena, 24, (dahl_fp*)&expect);

    dahl_block_p* block_p = block_partition_flatten_to_vector(block, DAHL_READ);

    dahl_vector const* flat_vector = GET_SUB_VECTOR(block_p, 0);
    ASSERT_SIZE_T_EQUALS(24, vector_get_len(flat_vector));
    ASSERT_VECTOR_EQUALS(expect_vector, flat_vector);

    block_unpartition(block_p);

    dahl_arena_reset(testing_arena);
}

void test_matrix_partition_along_y()
{
    dahl_shape2d data_shape = { .x = 4, .y = 5 };

    dahl_fp data[5][4] = {
        {-2, 1, 2,-1 },
        { 3, 1,-3, 1 },
        { 4,-1, 4,-1 },
        { 3, 1,-3, 1 },
        { 4,-1, 4,-1 },
    };

    dahl_matrix const* matrix = matrix_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    size_t expect_len = 4;

    dahl_matrix_p* matrix_p = matrix_partition_along_y(matrix, DAHL_READ);

    for (size_t i = 0; i < GET_NB_CHILDREN(matrix_p); i++)
    {
        dahl_vector const* sub_vector = GET_SUB_VECTOR(matrix_p, i);
        size_t len = vector_get_len(sub_vector);
        ASSERT_SIZE_T_EQUALS(expect_len, len);
    }

    matrix_unpartition(matrix_p);

    dahl_arena_reset(testing_arena);
}

void test_matrix_partition_along_y_batch()
{
    dahl_matrix* matrix = MATRIX(testing_arena, 6, 4, {
        {-2, 1, 2,-1 },
        { 3, 1,-3, 1 },
        { 8, 1,-3, 1 },
        { 4,-3, 4,-1 },
        { 8, 8,-5, 3 },
        { 4,-1, 9,-2 },
    });

    dahl_matrix* expect_matrices[3];

    expect_matrices[0] = MATRIX(testing_arena, 2, 4, {
        {-2, 1, 2,-1 },
        { 3, 1,-3, 1 },
    });

    expect_matrices[1] = MATRIX(testing_arena, 2, 4, {
        { 8, 1,-3, 1 },
        { 4,-3, 4,-1 },
    });

    expect_matrices[2] = MATRIX(testing_arena, 2, 4, {
        { 8, 8,-5, 3 },
        { 4,-1, 9,-2 },
    });

    dahl_matrix_p* matrix_p = matrix_partition_along_y_batch(matrix, DAHL_READ, 2);

    for (size_t i = 0; i < GET_NB_CHILDREN(matrix_p); i++)
    {
        dahl_matrix const* sub_matrix = GET_SUB_MATRIX(matrix_p, i);
        ASSERT_MATRIX_EQUALS(expect_matrices[i], sub_matrix);
    }

    matrix_unpartition(matrix_p);

    dahl_arena_reset(testing_arena);
}

void test_matrix_get_shape()
{
    dahl_fp data[3][4] = {
        { 3, 1,-8,-3 },
        {-7,-3, 3, 2 },
        { 1, 1, 9, 1 },
    };

    dahl_shape2d shape = { .x = 4, .y = 3 };

    dahl_matrix* matrix = matrix_init_from(testing_arena, shape, (dahl_fp*)&data);

    // Inject sleep on the matrix to verify that the resize task is really synchronous
    TASK_WAIT(matrix, 1000);
    task_matrix_as_flat_col(matrix);

    dahl_shape2d expect_shape = { .x = 1, .y = 12 };

    dahl_shape2d res_shape = matrix_get_shape(matrix);

    ASSERT_SHAPE2D_EQUALS(expect_shape, res_shape);

    dahl_arena_reset(testing_arena);
}

void test_recursive_partitioning()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2, 1, 2,-1 },
            { 3, 1,-3, 1 },
            { 4,-1, 4,-1 },
        },
        {
            { 3, 1,-8,-3 },
            {-7,-3, 3, 2 },
            { 1, 1, 9, 1 },
        },
    };

    dahl_block* block = block_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_vector* expect[2][3] = {
        {
            vector_init_from(testing_arena, 4, (dahl_fp[4]){-2, 1, 2,-1 }),
            vector_init_from(testing_arena, 4, (dahl_fp[4]){ 3, 1,-3, 1 }),
            vector_init_from(testing_arena, 4, (dahl_fp[4]){ 4,-1, 4,-1 }),
        },
        {
            vector_init_from(testing_arena, 4, (dahl_fp[4]){ 3, 1,-8,-3 }),
            vector_init_from(testing_arena, 4, (dahl_fp[4]){-7,-3, 3, 2 }),
            vector_init_from(testing_arena, 4, (dahl_fp[4]){ 1, 1, 9, 1 }),
        }
    };

    dahl_block_p* block_p = block_partition_along_z(block, DAHL_READ);

    for (size_t i = 0; i < GET_NB_CHILDREN(block_p); i++)
    {
        dahl_matrix const* matrix = GET_SUB_MATRIX(block_p, i);

        dahl_matrix_p* matrix_p = matrix_partition_along_y(matrix, DAHL_READ);

        for (size_t j = 0; j < GET_NB_CHILDREN(matrix_p); j++)
        {
            dahl_vector const* vector = GET_SUB_VECTOR(matrix_p, j);
            ASSERT_VECTOR_EQUALS(expect[i][j], vector);
        }

        matrix_unpartition(matrix_p);
    }

    block_unpartition(block_p);

    dahl_arena_reset(testing_arena);
}

void test_mut_partitioning()
{
    dahl_shape2d data_shape = { .x = 4, .y = 3 };

    dahl_fp data[3][4] = {
        {-2, 1, 2,-1 },
        { 3, 1,-3, 1 },
        { 4,-1, 4,-1 },
    };

    dahl_matrix* matrix = matrix_init_from(testing_arena, data_shape, (dahl_fp*)&data);
    dahl_matrix* another_matrix = matrix_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_matrix_p* matrix_p = matrix_partition_along_y(matrix, DAHL_READ);

    // Here I can still read from matrix because the partitionning is read only
    TASK_ADD_SELF(another_matrix, matrix);

    matrix_unpartition(matrix_p);

    matrix_p = matrix_partition_along_y(matrix, DAHL_MUT);
    // Here I cannot read the matrix handle because it is mutably partioned
    // TASK_ADD_SELF(another_matrix, matrix);
    // TODO: I should be able to test something that should fail

    matrix_unpartition(matrix_p);

    dahl_arena_reset(testing_arena);
}

void test_partition_reuse()
{
    dahl_shape2d data_shape = { .x = 4, .y = 3 };

    dahl_fp data[3][4] = {
        {-2, 1, 2,-1 },
        { 3, 1,-3, 1 },
        { 4,-1, 4,-1 },
    };

    dahl_fp expect[3][4] = {
        {-8, 4, 8,-4 },
        { 3, 1,-3, 1 },
        { 4,-1, 4,-1 },
    };

    dahl_matrix* matrix = matrix_init_from(testing_arena, data_shape, (dahl_fp*)&data);
    dahl_matrix* expect_matrix = matrix_init_from(testing_arena, data_shape, (dahl_fp*)&expect);

    dahl_matrix_p* matrix_p = matrix_partition_along_y(matrix, DAHL_MUT);

    dahl_vector* vector = GET_SUB_VECTOR_MUT(matrix_p, 0);
    TASK_SCAL_SELF(vector, 2);
    matrix_unpartition(matrix_p);

    REACTIVATE_PARTITION(matrix_p);
    vector = GET_SUB_VECTOR_MUT(matrix_p, 0);
    TASK_SCAL_SELF(vector, 2);
    matrix_unpartition(matrix_p);

    ASSERT_MATRIX_EQUALS(expect_matrix, matrix);
    dahl_arena_reset(testing_arena);
}

void test_tensor_flatten_along_t_no_copy()
{
    dahl_tensor* tensor = TENSOR(testing_arena, 2, 1, 4, 3, {
        {{
                { 0, 1, 2 },
                { 0, 1, 2 },
                { 0, 1, 2 },
                { 0, 1, 2 },
        }},
        {{
                { 2, 4, 6 },
                { 2, 4, 6 },
                { 2, 4, 6 },
                { 2, 4, 6 },
        }},
    });

    dahl_matrix* expect = MATRIX(testing_arena, 2, 12, {
        { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 }, 
        { 2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6 }
    });

    dahl_matrix* matrix = tensor_flatten_along_t_no_copy(tensor);

    ASSERT_MATRIX_EQUALS(expect, matrix);
    dahl_arena_reset(testing_arena);
}

void test_matrix_to_tensor_no_copy()
{
    dahl_matrix* matrix = MATRIX(testing_arena, 2, 12, {
        { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 }, 
        { 2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6 }
    });

    dahl_tensor* expect = TENSOR(testing_arena, 2, 1, 4, 3, {
        {{
                { 0, 1, 2 },
                { 0, 1, 2 },
                { 0, 1, 2 },
                { 0, 1, 2 },
        }},
        {{
                { 2, 4, 6 },
                { 2, 4, 6 },
                { 2, 4, 6 },
                { 2, 4, 6 },
        }},
    });

    dahl_shape4d shape = { .x = 3, .y = 4, .z = 1, .t = 2 };
    dahl_tensor* tensor = matrix_to_tensor_no_copy(matrix, shape);

    ASSERT_TENSOR_EQUALS(expect, tensor);
    dahl_arena_reset(testing_arena);
}

void test_block_read_jpeg()
{
    dahl_block* a = block_init(testing_arena, (dahl_shape3d){ .x = 1080, .y = 1440, .z = 3});
    block_read_jpeg(a, "../datasets/big-fashion/images/1163.jpg"); 

    block_image_display(a, 1);
}

void test_data()
{
    test_tensor_partition_along_t();
    test_tensor_partition_along_t_batch();
    test_block_partition_along_z();
    test_block_partition_flatten_to_vector();
    test_matrix_partition_along_y();
    test_matrix_partition_along_y_batch();
    test_matrix_get_shape();
    test_recursive_partitioning();
    test_mut_partitioning();
    test_partition_reuse();
    test_tensor_flatten_along_t_no_copy();
    test_matrix_to_tensor_no_copy();
    test_block_read_jpeg();
}
