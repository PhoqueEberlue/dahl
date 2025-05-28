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

    block_partition_along_z(block);

    for (size_t i = 0; i < block_get_sub_matrix_nb(block); i++)
    {
        dahl_matrix* sub_matrix = block_get_sub_matrix(block, i);
        dahl_shape2d shape = matrix_get_shape(sub_matrix);
        ASSERT_SHAPE2D_EQUALS(expect_shape, shape);
    }

    block_unpartition(block);

    block_finalize(block);
}

void test_matrix_partition_along_y()
{
    dahl_shape2d data_shape = { .x = 4, .y = 3 };

    dahl_fp data[3][4] = {
        {-2.0F, 1.0F, 2.0F,-1.0F },
        { 3.0F, 1.0F,-3.0F, 1.0F },
        { 4.0F,-1.0F, 4.0F,-1.0F },
    };

    dahl_matrix* matrix = matrix_init_from(data_shape, (dahl_fp*)&data);

    size_t expect_len = 4;

    matrix_partition_along_y(matrix);

    for (size_t i = 0; i < matrix_get_sub_vector_nb(matrix); i++)
    {
        dahl_vector* sub_vector = matrix_get_sub_vector(matrix, i);
        size_t len = vector_get_len(sub_vector);
        ASSERT_SIZE_T_EQUALS(expect_len, len);
    }

    matrix_unpartition(matrix);

    matrix_finalize(matrix);
}


void test_block_to_vector()
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

    dahl_vector* vec = block_to_vector(block);

    dahl_fp res = task_vector_dot_product(vec, vec);

    ASSERT_FP_EQUALS(res, 302.0F);

    vector_finalize(vec);
    // Here no need to finalize the block
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

    ASSERT_BLOCK_EQUALS(expect_block, padded_block, false);

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

    ASSERT_BLOCK_EQUALS(expect_block_2, padded_block_2, false);
}


void test_data()
{
    test_block_partition_along_z();
    test_matrix_partition_along_y();
    test_block_to_vector();
    test_block_add_padding();
}
