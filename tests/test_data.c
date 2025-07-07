#include "tests.h"
#include <stdio.h>

void test_block_partition_along_z()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2.0F, 1.0F, 2.0F,-1.0F },
            { 3.0F, 1.0F,-3.0F, 1.0F },
            { 4.0F,-1.0F, 4.0F,-1.0F },
        },
        {
            { 3.0F, 1.0F,-8.0F,-3.0F },
            {-7.0F,-3.0F, 3.0F, 2.0F },
            { 1.0F, 1.0F, 9.0F, 1.0F },
        },
    };

    dahl_block* block = block_init_from(data_shape, (dahl_fp*)&data);

    dahl_shape2d expect_shape = { .x = 4, .y = 3 };

    dahl_fp expect_0[3][4] = {
        {-2.0F, 1.0F, 2.0F,-1.0F },
        { 3.0F, 1.0F,-3.0F, 1.0F },
        { 4.0F,-1.0F, 4.0F,-1.0F },
    };

    dahl_fp expect_1[3][4] = {
        { 3.0F, 1.0F,-8.0F,-3.0F },
        {-7.0F,-3.0F, 3.0F, 2.0F },
        { 1.0F, 1.0F, 9.0F, 1.0F },
    };

    dahl_matrix* expect_matrix_0 = matrix_init_from(expect_shape, (dahl_fp*)&expect_0);
    dahl_matrix* expect_matrix_1 = matrix_init_from(expect_shape, (dahl_fp*)&expect_1);

    block_partition_along_z(block);

    dahl_matrix* sub_matrix_0 = block_get_sub_matrix(block, 0);
    dahl_shape2d shape_0 = matrix_get_shape(sub_matrix_0);
    ASSERT_SHAPE2D_EQUALS(expect_shape, shape_0);
    ASSERT_MATRIX_EQUALS(expect_matrix_0, sub_matrix_0);

    dahl_matrix* sub_matrix_1 = block_get_sub_matrix(block, 1);
    dahl_shape2d shape_1 = matrix_get_shape(sub_matrix_1);
    ASSERT_SHAPE2D_EQUALS(expect_shape, shape_1);
    ASSERT_MATRIX_EQUALS(expect_matrix_1, sub_matrix_1);

    block_unpartition(block);

    dahl_arena_reset(testing_arena);
}

void test_block_partition_flatten_to_vector()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2.0F, 1.0F, 2.0F,-1.0F },
            { 3.0F, 1.0F,-3.0F, 1.0F },
            { 4.0F,-1.0F, 4.0F,-1.0F },
        },
        {
            { 3.0F, 1.0F,-8.0F,-3.0F },
            {-7.0F,-3.0F, 3.0F, 2.0F },
            { 1.0F, 1.0F, 9.0F, 1.0F },
        },
    };

    dahl_block* block = block_init_from(data_shape, (dahl_fp*)&data);

    dahl_fp expect[24] = {
        -2.0F, 1.0F, 2.0F,-1.0F,
        3.0F, 1.0F,-3.0F, 1.0F,
        4.0F,-1.0F, 4.0F,-1.0F,
        3.0F, 1.0F,-8.0F,-3.0F,
       -7.0F,-3.0F, 3.0F, 2.0F,
        1.0F, 1.0F, 9.0F, 1.0F,
    };

    dahl_vector* expect_vector = vector_init_from(24, (dahl_fp*)&expect);

    block_partition_flatten_to_vector(block);

    dahl_vector* flat_vector = block_get_sub_vector(block, 0);
    ASSERT_SIZE_T_EQUALS(24, vector_get_len(flat_vector));
    ASSERT_VECTOR_EQUALS(expect_vector, flat_vector);

    block_unpartition(block);

    dahl_arena_reset(testing_arena);
}

void test_matrix_partition_along_y()
{
    dahl_shape2d data_shape = { .x = 4, .y = 5 };

    dahl_fp data[5][4] = {
        {-2.0F, 1.0F, 2.0F,-1.0F },
        { 3.0F, 1.0F,-3.0F, 1.0F },
        { 4.0F,-1.0F, 4.0F,-1.0F },
        { 3.0F, 1.0F,-3.0F, 1.0F },
        { 4.0F,-1.0F, 4.0F,-1.0F },
    };

    dahl_matrix* matrix = matrix_init_from(data_shape, (dahl_fp*)&data);

    size_t expect_len = 4;

    matrix_partition_along_y(matrix);

    for (size_t i = 0; i < matrix_get_nb_children(matrix); i++)
    {
        dahl_vector* sub_vector = matrix_get_sub_vector(matrix, i);
        size_t len = vector_get_len(sub_vector);
        ASSERT_SIZE_T_EQUALS(expect_len, len);
    }

    matrix_unpartition(matrix);

    dahl_arena_reset(testing_arena);
}

void test_block_add_padding()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2.0F, 1.0F, 2.0F,-1.0F },
            { 3.0F, 1.0F,-3.0F, 1.0F },
            { 4.0F,-1.0F, 4.0F,-1.0F },
        },
        {
            { 3.0F, 1.0F,-8.0F,-3.0F },
            {-7.0F,-3.0F, 3.0F, 2.0F },
            { 1.0F, 1.0F, 9.0F, 1.0F },
        },
    };

    dahl_fp expect[4][5][6] = {
        {
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
        },
        {
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F,-2.000000F, 1.000000F, 2.000000F,-1.000000F, 0.000000F }, 
            { 0.000000F, 3.000000F, 1.000000F,-3.000000F, 1.000000F, 0.000000F }, 
            { 0.000000F, 4.000000F,-1.000000F, 4.000000F,-1.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
        },
        {
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 3.000000F, 1.000000F,-8.000000F,-3.000000F, 0.000000F }, 
            { 0.000000F,-7.000000F,-3.000000F, 3.000000F, 2.000000F, 0.000000F }, 
            { 0.000000F, 1.000000F, 1.000000F, 9.000000F, 1.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
        },
        {
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F }, 
        },
    };

    dahl_block* block = block_init_from(data_shape, (dahl_fp*)&data);

    dahl_shape3d padded_shape = { .x = 6, .y = 5, .z = 4 };
    dahl_block* padded_block = block_add_padding_init(block, padded_shape);

    dahl_block* expect_block = block_init_from(padded_shape, (dahl_fp*)&expect);

    ASSERT_BLOCK_EQUALS(expect_block, padded_block);

    dahl_shape3d padded_shape_2 = { .x = 8, .y = 7, .z = 2 };
    dahl_block* padded_block_2 = block_add_padding_init(block, padded_shape_2);

    dahl_fp expect_2[2][7][8] = {
        {
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F,-2.000000F, 1.000000F, 2.000000F,-1.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 3.000000F, 1.000000F,-3.000000F, 1.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 4.000000F,-1.000000F, 4.000000F,-1.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
        },
        {
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 3.000000F, 1.000000F,-8.000000F,-3.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F,-7.000000F,-3.000000F, 3.000000F, 2.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 1.000000F, 1.000000F, 9.000000F, 1.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
            { 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, 0.000000F, }, 
        },
    };

    dahl_block* expect_block_2 = block_init_from(padded_shape_2, (dahl_fp*)&expect_2);

    ASSERT_BLOCK_EQUALS(expect_block_2, padded_block_2);

    dahl_arena_reset(testing_arena);
}

void test_matrix_get_shape()
{
    dahl_fp data[3][4] = {
        { 3.0F, 1.0F,-8.0F,-3.0F },
        {-7.0F,-3.0F, 3.0F, 2.0F },
        { 1.0F, 1.0F, 9.0F, 1.0F },
    };

    dahl_shape2d shape = { .x = 4, .y = 3 };

    dahl_matrix* matrix = matrix_init_from(shape, (dahl_fp*)&data);

    // Inject sleep on the matrix to verify that the resize task is really synchronous
    TASK_WAIT(matrix, 1000);
    task_matrix_to_flat_col(matrix);

    dahl_shape2d expect_shape = { .x = 1, .y = 12 };

    dahl_shape2d res_shape = matrix_get_shape(matrix);

    ASSERT_SHAPE2D_EQUALS(expect_shape, res_shape);

    dahl_arena_reset(testing_arena);
}

void test_recursive_partitionning()
{
    dahl_shape3d data_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp data[2][3][4] = {
        {
            {-2.0F, 1.0F, 2.0F,-1.0F },
            { 3.0F, 1.0F,-3.0F, 1.0F },
            { 4.0F,-1.0F, 4.0F,-1.0F },
        },
        {
            { 3.0F, 1.0F,-8.0F,-3.0F },
            {-7.0F,-3.0F, 3.0F, 2.0F },
            { 1.0F, 1.0F, 9.0F, 1.0F },
        },
    };

    dahl_block* block = block_init_from(data_shape, (dahl_fp*)&data);

    dahl_vector* expect[2][3] = {
        {
            vector_init_from(4, (dahl_fp[4]){-2.0F, 1.0F, 2.0F,-1.0F }),
            vector_init_from(4, (dahl_fp[4]){ 3.0F, 1.0F,-3.0F, 1.0F }),
            vector_init_from(4, (dahl_fp[4]){ 4.0F,-1.0F, 4.0F,-1.0F }),
        },
        {
            vector_init_from(4, (dahl_fp[4]){ 3.0F, 1.0F,-8.0F,-3.0F }),
            vector_init_from(4, (dahl_fp[4]){-7.0F,-3.0F, 3.0F, 2.0F }),
            vector_init_from(4, (dahl_fp[4]){ 1.0F, 1.0F, 9.0F, 1.0F }),
        }
    };

    block_partition_along_z(block);

    for (size_t i = 0; i < block_get_nb_children(block); i++)
    {
        dahl_matrix* matrix = block_get_sub_matrix(block, i);

        matrix_partition_along_y(matrix);

        for (size_t j = 0; j < matrix_get_nb_children(matrix); j++)
        {
            dahl_vector* vector = matrix_get_sub_vector(matrix, j);
            ASSERT_VECTOR_EQUALS(expect[i][j], vector);
        }

        matrix_unpartition(matrix);
    }

    block_unpartition(block);
}

void test_data()
{
    test_block_partition_along_z();
    test_block_partition_flatten_to_vector();
    test_matrix_partition_along_y();
    test_block_add_padding();
    test_matrix_get_shape();
    test_recursive_partitionning();
}
