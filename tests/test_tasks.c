#include "tests.h"
#include "../src/tasks.h"
#include <math.h>
#include <stdio.h>

// Asserts that "a cross correlation b = expect"
void assert_cross_correlation_2d(dahl_fp* a, shape2d a_shape,
                                 dahl_fp* b, shape2d b_shape,
                                 dahl_fp* expect, shape2d expect_shape)
{
    dahl_matrix* a_matrix = matrix_init_from(a_shape, a);
    dahl_matrix* b_matrix = matrix_init_from(b_shape, b);
    dahl_matrix* c_matrix = matrix_init(expect_shape);
    dahl_matrix* expect_matrix = matrix_init_from(expect_shape, expect);

    task_matrix_cross_correlation(a_matrix, b_matrix, c_matrix);

    starpu_task_wait_for_all();

    assert(matrix_equals(expect_matrix, c_matrix));

    matrix_free(a_matrix);
    matrix_free(b_matrix);
    matrix_free(c_matrix);
    matrix_free(expect_matrix);
}

void test_cross_correlation_2d_1()
{
    shape2d a_shape = { .x = 5, .y = 5 };
    shape2d b_shape = { .x = 3, .y = 3 };
    shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

    dahl_fp a[5][5] = {
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
    };

    dahl_fp b[3][3] = {
        { 1.0F, 0.0F, 1.0F },
        { 0.0F, 1.0F, 0.0F },
        { 1.0F, 0.0F, 1.0F },
    };

    dahl_fp expect[3][3] = {
        { 5.0F, 5.0F, 5.0F },
        { 5.0F, 5.0F, 5.0F },
        { 5.0F, 5.0F, 5.0F },
    };

    assert_cross_correlation_2d((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);
}

void test_cross_correlation_2d_2()
{
    shape2d a_shape = { .x = 7, .y = 5 };
    shape2d b_shape = { .x = 4, .y = 3 };
    shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

    dahl_fp a[5][7] = {
        { 0.0F, 1.0F, 0.0F, 4.0F, 0.0F, 3.0F, 2.0F },
        { 0.0F, 0.0F, 6.0F, 0.0F, 8.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 0.0F, 0.0F, 7.0F, 1.0F, 0.0F },
        { 0.0F, 0.0F, 2.0F, 1.0F, 0.0F, 1.0F, 1.0F },
        { 8.0F, 9.0F, 0.0F, 2.0F, 3.0F, 0.0F, 0.0F },
    };

    dahl_fp b[3][4] = {
        { 2.0F, 1.0F, 2.0F, 1.0F },
        { 3.0F, 1.0F, 3.0F, 1.0F },
        { 4.0F, 1.0F, 4.0F, 1.0F },
    };

    dahl_fp expect[3][4] = {
        { 28.0F, 35.0F, 79.0F, 39.0F }, 
        { 25.0F, 30.0F, 61.0F, 30.0F }, 
        { 53.0F, 61.0F, 37.0F, 27.0F },
    };

    assert_cross_correlation_2d((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);
}

void test_relu()
{
    shape3d a_shape = { .x = 4, .y = 3, .z = 2 };
    shape3d expect_shape = a_shape;

    dahl_fp a[2][3][4] = {
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

    dahl_fp expect[2][3][4] = {
        {
            { 0.0F, 1.0F, 2.0F, 0.0F },
            { 3.0F, 1.0F, 0.0F, 1.0F },
            { 4.0F, 0.0F, 4.0F, 0.0F },
        },
        {
            { 3.0F, 1.0F, 0.0F, 0.0F },
            { 0.0F, 0.0F, 3.0F, 2.0F },
            { 1.0F, 1.0F, 9.0F, 1.0F },
        },
    };

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    task_block_relu(a_block);

    assert(block_equals(expect_block, a_block));
    block_free(a_block);
    block_free(expect_block);
}

void test_sum_z_axis()
{
    shape3d a_shape = { .x = 4, .y = 3, .z = 2 };
    shape2d expect_shape = { .x = 4, .y = 3 };

    dahl_fp a[2][3][4] = {
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

    dahl_fp expect[3][4] = {
        { 1.0F, 2.0F,-6.0F,-4.0F },
        {-4.0F,-2.0F, 0.0F, 3.0F },
        { 5.0F, 0.0F,13.0F, 0.0F },
    };

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_matrix* expect_matrix = matrix_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_matrix* result_matrix = task_block_sum_z_axis(a_block);

    assert(matrix_equals(expect_matrix, result_matrix));
    block_free(a_block);
    matrix_free(expect_matrix);
    matrix_free(result_matrix);
}

void test_scal()
{
    shape3d a_shape = { .x = 4, .y = 3, .z = 2 };
    shape3d expect_shape = a_shape;

    dahl_fp a[2][3][4] = {
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

    dahl_fp expect[2][3][4] = {
        {
            {-4.0F, 2.0F, 4.0F,-2.0F },
            { 6.0F, 2.0F,-6.0F, 2.0F },
            { 8.0F,-2.0F, 8.0F,-2.0F },
        },
        {
            { 6.0F, 2.0F,-16.0F,-6.0F },
            {-14.0F,-6.0F, 6.0F, 4.0F },
            { 2.0F, 2.0F, 18.0F, 2.0F },
        },
    };

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    task_block_scal_self(a_block, 2);

    assert(block_equals(expect_block, a_block));
    block_free(a_block);
    block_free(expect_block);
}

void test_sub()
{
    shape3d a_shape = { .x = 2, .y = 2, .z = 2 };
    shape3d b_shape = a_shape;
    shape3d expect_shape = a_shape;

    dahl_fp a[2][2][2] = {
        {
            {-2.0F, 1.0F },
            { 3.0F, 1.0F },
        },
        {
            { 3.0F, 1.0F },
            {-7.0F,-3.0F },
        },
    };

    dahl_fp b[2][2][2] = {
        {
            { 9.0F, 9.0F },
            {-7.0F, 3.0F },
        },
        {
            {-2.0F, 4.0F },
            {-6.0F, 0.0F },
        },
    };

    dahl_fp expect[2][2][2] = {
        {
            {-11.0F,-8.0F },
            { 10.0F,-2.0F },
        },
        {
            { 5.0F,-3.0F },
            {-1.0F,-3.0F },
        },
    };

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* b_block = block_init_from(b_shape, (dahl_fp*)&b);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_block* result_block = task_block_sub(a_block, b_block);

    assert(block_equals(expect_block, result_block)); 

    // here it modifies a instead of returning the result
    task_block_sub_self(a_block, b_block);
    assert(block_equals(expect_block, a_block));

    block_free(a_block);
    block_free(b_block);
    block_free(result_block);
    block_free(expect_block);
}

void test_add()
{
    shape3d a_shape = { .x = 2, .y = 2, .z = 2 };
    shape3d b_shape = a_shape;
    shape3d expect_shape = a_shape;

    dahl_fp a[2][2][2] = {
        {
            {-2.0F, 1.0F },
            { 3.0F, 1.0F },
        },
        {
            { 3.0F, 1.0F },
            {-7.0F,-3.0F },
        },
    };

    dahl_fp b[2][2][2] = {
        {
            { 9.0F, 9.0F },
            {-7.0F, 3.0F },
        },
        {
            {-2.0F, 4.0F },
            {-6.0F, 0.0F },
        },
    };

    dahl_fp expect[2][2][2] = {
        {
            { 7.0F, 10.0F },
            {-4.0F, 4.0F },
        },
        {
            { 1.0F, 5.0F },
            {-13.0F,-3.0F },
        },
    };

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* b_block = block_init_from(b_shape, (dahl_fp*)&b);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_block* result_block = task_block_add(a_block, b_block);

    assert(block_equals(expect_block, result_block)); 

    // here it modifies a instead of returning the result
    task_block_add_self(a_block, b_block);
    assert(block_equals(expect_block, a_block));

    block_free(a_block);
    block_free(b_block);
    block_free(result_block);
    block_free(expect_block);
}

void test_tasks()
{
    test_cross_correlation_2d_1();
    test_cross_correlation_2d_2();
    test_relu();
    test_sum_z_axis();
    test_scal();
    test_sub();
    test_add();
}
