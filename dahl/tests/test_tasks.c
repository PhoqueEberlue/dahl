#include "test_layers_data.h"
#include "tests.h"
#include <assert.h>
#include <stdio.h>
#include "../src/data_structures/data_structures.h"

// Asserts that "a cross correlation b = expect"
void assert_matrix_cross_correlation(dahl_fp* a, dahl_shape2d a_shape,
                                     dahl_fp* b, dahl_shape2d b_shape,
                                     dahl_fp* expect, dahl_shape2d expect_shape)
{
    dahl_matrix* a_matrix = matrix_init_from(testing_arena, a_shape, a);
    dahl_matrix* b_matrix = matrix_init_from(testing_arena, b_shape, b);
    dahl_matrix* c_matrix = matrix_init(testing_arena, expect_shape);
    dahl_matrix* expect_matrix = matrix_init_from(testing_arena, expect_shape, expect);

    task_matrix_cross_correlation(a_matrix, b_matrix, c_matrix);

    ASSERT_MATRIX_EQUALS(expect_matrix, c_matrix);

    dahl_arena_reset(testing_arena);
}

void test_matrix_cross_correlation_1()
{
    dahl_shape2d a_shape = { .x = 5, .y = 5 };
    dahl_shape2d b_shape = { .x = 3, .y = 3 };
    dahl_shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

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

    assert_matrix_cross_correlation((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);
}

void test_matrix_cross_correlation_2()
{
    dahl_shape2d a_shape = { .x = 7, .y = 5 };
    dahl_shape2d b_shape = { .x = 4, .y = 3 };
    dahl_shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

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

    assert_matrix_cross_correlation((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);

    dahl_arena_reset(testing_arena);
}

void test_relu()
{
    dahl_block* block = BLOCK(testing_arena, 2, 3, 4, {
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
    });

    dahl_block* expect_block = BLOCK(testing_arena, 2, 3, 4, {
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
    });

    TASK_RELU_SELF(block);
    ASSERT_BLOCK_EQUALS(expect_block, block);

    dahl_matrix* matrix = MATRIX(testing_arena, 3, 4, {
        { 3.0F, 1.0F,-8.0F,-3.0F },
        {-7.0F,-3.0F, 3.0F, 2.0F },
        { 1.0F, 1.0F, 9.0F, 1.0F },
    });

    dahl_matrix* expect_matrix = MATRIX(testing_arena, 3, 4, {
        { 3.0F, 1.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 3.0F, 2.0F },
        { 1.0F, 1.0F, 9.0F, 1.0F },
    });

    TASK_RELU_SELF(matrix);
    ASSERT_MATRIX_EQUALS(expect_matrix, matrix);

    dahl_vector* vector = VECTOR(testing_arena, 4, { 3.0F, 1.0F,-8.0F,-3.0F });
    dahl_vector* out_vector = vector_init(testing_arena, 4);
    dahl_vector* expect_vector = VECTOR(testing_arena, 4, { 3.0F, 1.0F, 0.0F, 0.0F });

    TASK_RELU(vector, out_vector);
    ASSERT_VECTOR_EQUALS(expect_vector, out_vector);

    dahl_arena_reset(testing_arena);
}

void test_relu_backward()
{
    dahl_block* input = BLOCK(testing_arena, 2, 3, 4, {
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
    });

    dahl_block* gradients = BLOCK(testing_arena, 2, 3, 4, {
        {
            { 0.5F, 0.5F, 0.2F, 0.4F },
            { 0.8F, 4.8F, 7.0F, 6.7F },
            { 4.9F, 7.7F, 6.0F, 6.0F },
        },
        {
            { 0.5F, 5.0F, 9.3F, 1.0F },
            { 7.9F, 9.8F, 8.4F, 0.6F },
            { 4.5F, 0.8F, 7.7F, 8.2F },
        },
    });

    dahl_block* expect = BLOCK(testing_arena, 2, 3, 4, {
        {
            { 0.0F, 0.5F, 0.2F, 0.0F },
            { 0.8F, 4.8F, 0.0F, 6.7F },
            { 4.9F, 0.0F, 6.0F, 0.0F },
        },
        {
            { 0.5F, 5.0F, 0.0F, 0.0F },
            { 0.0F, 0.0F, 8.4F, 0.6F },
            { 4.5F, 0.8F, 7.7F, 8.2F },
        },
    });

    dahl_block* out = block_init(testing_arena, (dahl_shape3d){ .x = 4, .y = 3, .z = 2 });
    TASK_RELU_BACKWARD(input, gradients, out);
    ASSERT_BLOCK_EQUALS(expect, out);

    dahl_arena_reset(testing_arena);
}

void test_tensor_sum_t_axis()
{
    dahl_shape4d a_shape = { .x = 4, .y = 3, .z = 2, .t = 2 };
    dahl_shape3d expect_shape = { .x = 4, .y = 3, .z = 2 };

    dahl_fp a[2][2][3][4] = {
        {
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
        },
        {
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
        }
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
        }
    };

    dahl_tensor* a_tensor = tensor_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_block* expect_block = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);
    dahl_block* result_block = block_init(testing_arena, expect_shape);
    task_tensor_sum_t_axis(a_tensor, result_block);

    ASSERT_BLOCK_EQUALS(expect_block, result_block);

    dahl_arena_reset(testing_arena);
}

void test_tensor_sum_xyt_axis()
{
    dahl_tensor* in = TENSOR(testing_arena, 2, 2, 3, 4, {
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
                {-2, 1, 2,-1 },
                { 3, 1,-3, 1 },
                { 4,-1, 4,-1 },
            },
            {
                { 3, 1,-8,-3 },
                {-7,-3, 3, 2 },
                { 1, 1, 9, 1 },
            },
        }
    });

    dahl_vector* expect = VECTOR(testing_arena, 2, { 16, 0 });

    dahl_vector* out = task_tensor_sum_xyt_axes_init(testing_arena, in);
    ASSERT_VECTOR_EQUALS(expect, out);

    dahl_arena_reset(testing_arena);
}

void test_block_sum_z_axis()
{
    dahl_shape3d a_shape = { .x = 4, .y = 3, .z = 2 };
    dahl_shape2d expect_shape = { .x = 4, .y = 3 };

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

    dahl_block* a_block = block_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_matrix* expect_matrix = matrix_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);
    dahl_matrix* result_matrix = matrix_init(testing_arena, expect_shape);
    task_block_sum_z_axis(a_block, result_matrix);

    ASSERT_MATRIX_EQUALS(expect_matrix, result_matrix);

    dahl_arena_reset(testing_arena);
}

void test_matrix_sum_y_axis()
{
    dahl_shape2d a_shape = { .x = 4, .y = 3 };
    size_t expect_len = 4;

    dahl_fp a[3][4] = {
        {-2.0F, 1.0F, 2.0F,-1.0F },
        { 3.0F, 1.0F,-3.0F, 1.0F },
        { 4.0F,-1.0F, 4.0F,-1.0F },
    };

    dahl_fp expect[4] = { 5.0F, 1.0F, 3.0F,-1.0F };

    dahl_matrix* a_matrix = matrix_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_vector* expect_vector = vector_init_from(testing_arena, expect_len, (dahl_fp*)&expect);

    dahl_vector* result_vector = task_matrix_sum_y_axis_init(testing_arena, a_matrix);

    ASSERT_VECTOR_EQUALS(expect_vector, result_vector);

    dahl_arena_reset(testing_arena);
}

void test_scal()
{
    dahl_shape3d a_shape = { .x = 4, .y = 3, .z = 2 };
    dahl_shape3d expect_shape = a_shape;

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

    dahl_block* a_block = block_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_block* expect_block = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);

    TASK_SCAL_SELF(a_block, 2);

    ASSERT_BLOCK_EQUALS(expect_block, a_block);

    dahl_arena_reset(testing_arena);
}

void test_power()
{
    dahl_vector* in = VECTOR(testing_arena, 4, { 1, 2, 3, 4 });
    dahl_vector* expect = VECTOR(testing_arena, 4, { 1, 4, 9, 16 });

    TASK_POWER_SELF(in, 2);
    ASSERT_VECTOR_EQUALS(expect, in);

    expect = VECTOR(testing_arena, 4, { 1, 1, 1, 1 });
    TASK_POWER_SELF(in, 0);
    ASSERT_VECTOR_EQUALS(expect, in);

    dahl_arena_reset(testing_arena);
}

void test_divide()
{
    dahl_shape3d a_shape = { .x = 4, .y = 3, .z = 2 };
    dahl_shape3d expect_shape = a_shape;

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
            {-1.0F, 0.5F, 1.0F,-0.5F },
            { 1.5F, 0.5F,-1.5F, 0.5F },
            { 2.0F,-0.5F, 2.0F,-0.5F },
        },
        {
            { 1.5F, 0.5F,-4.0F,-1.5F },
            {-3.5F,-1.5F, 1.5F, 1.0F },
            { 0.5F, 0.5F, 4.5F, 0.5F },
        },
    };

    dahl_block* a_block = block_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_block* expect_block = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);

    TASK_DIVIDE_SELF(a_block, 2e0);
    ASSERT_BLOCK_EQUALS(expect_block, a_block);

    dahl_arena_reset(testing_arena);
}

void test_sub()
{
    dahl_shape3d a_shape = { .x = 2, .y = 2, .z = 2 };
    dahl_shape3d b_shape = a_shape;
    dahl_shape3d expect_shape = a_shape;

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

    dahl_block* a_block = block_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_block* b_block = block_init_from(testing_arena, b_shape, (dahl_fp*)&b);
    dahl_block* expect_block = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);

    dahl_block* result_block = block_init(testing_arena, a_shape);
    TASK_SUB(a_block, b_block, result_block);

    ASSERT_BLOCK_EQUALS(expect_block, result_block); 

    // here it modifies `a` instead of returning the result
    TASK_SUB_SELF(a_block, b_block);
    ASSERT_BLOCK_EQUALS(expect_block, a_block);

    dahl_arena_reset(testing_arena);
}

void test_add()
{
    dahl_shape3d a_shape = { .x = 2, .y = 2, .z = 2 };
    dahl_shape3d b_shape = a_shape;
    dahl_shape3d expect_shape = a_shape;

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

    dahl_block* a_block = block_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_block* b_block = block_init_from(testing_arena, b_shape, (dahl_fp*)&b);
    dahl_block* expect_block = block_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);

    dahl_block* result_block = block_init(testing_arena, a_shape);
    TASK_ADD(a_block, b_block, result_block);

    ASSERT_BLOCK_EQUALS(expect_block, result_block); 

    // here it modifies a instead of returning the result
    TASK_ADD_SELF(a_block, b_block);
    ASSERT_BLOCK_EQUALS(expect_block, a_block);

    dahl_arena_reset(testing_arena);
}

void test_vector_softmax()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F };
    dahl_vector* in = vector_init_from(testing_arena, len, (dahl_fp*)&data);
    dahl_vector* out = vector_init(testing_arena, len);

    dahl_fp expect[len] = { 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F };
    dahl_vector* expect_vec = vector_init_from(testing_arena, len, (dahl_fp*)&expect);

    task_vector_softmax(in, out);

    // Note that values are rounded up in order to compare
    ASSERT_VECTOR_EQUALS_ROUND(expect_vec, out, 6);

    dahl_fp data_2[len] = { 1.8F, 3.8F, 8.7F, 6.9F, 3.9F, 12.9F, 6.0F, 3.7F, 6.1F, 3.2F };
    dahl_vector* in_2 = vector_init_from(testing_arena, len, (dahl_fp*)&data_2);

    dahl_fp expect_2[len] = { 0.000015F, 0.000109F, 0.014701F, 0.002430F, 0.000121F, 
                              0.980384F, 0.000988F, 0.000099F, 0.001092F, 0.000060F };
    dahl_vector* expect_vec_2 = vector_init_from(testing_arena, len, (dahl_fp*)&expect_2);

    task_vector_softmax(in_2, out);

    ASSERT_VECTOR_EQUALS_ROUND(expect_vec_2, out, 6);

    dahl_arena_reset(testing_arena);
}

void test_vector_dot_product()
{
    size_t constexpr len = 10;
    dahl_fp data_1[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* a = vector_init_from(testing_arena, len, (dahl_fp*)&data_1);

    dahl_fp expect = 285.0F;

    dahl_scalar* result = task_vector_dot_product_init(testing_arena, a, a);

    ASSERT_FP_EQUALS(expect, scalar_get_value(result));

    dahl_fp data_2[len] = { 9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.0F };
    dahl_vector* b = vector_init_from(testing_arena, len, (dahl_fp*)&data_2);

    dahl_fp expect_2 = 120.0F;

    result = task_vector_dot_product_init(testing_arena, a, b);

    ASSERT_FP_EQUALS(expect_2, scalar_get_value(result));

    dahl_arena_reset(testing_arena);
}

void test_vector_diag()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* a = vector_init_from(testing_arena, len, (dahl_fp*)&data);

    dahl_shape2d expect_shape = {.x=len, .y=len};
    dahl_fp expect[len][len] = { 
        { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 7.0F, 0.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 8.0F, 0.0F },
        { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 9.0F }
    };
    dahl_matrix* expect_matrix = matrix_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);

    dahl_matrix* result = task_vector_diag_init(testing_arena, a);

    ASSERT_MATRIX_EQUALS(expect_matrix, result);

    dahl_arena_reset(testing_arena);
}

void test_add_value()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* in = vector_init_from(testing_arena, len, (dahl_fp*)&data);

    dahl_vector* out = vector_init(testing_arena, len);

    dahl_fp expect[len] = { 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F };
    dahl_vector* expect_vec = vector_init_from(testing_arena, len, (dahl_fp*)&expect);

    TASK_ADD_VALUE(in, out, 4.0F);

    ASSERT_VECTOR_EQUALS(expect_vec, out);

    // Directly modifies in
    TASK_ADD_VALUE_SELF(in, 4.0F);

    ASSERT_VECTOR_EQUALS(expect_vec, in);

    dahl_arena_reset(testing_arena);
}

void test_sub_value()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* in = vector_init_from(testing_arena, len, (dahl_fp*)&data);

    dahl_vector* out = vector_init(testing_arena, len);

    dahl_fp expect[len] = { -4.0F, -3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F };
    dahl_vector* expect_vec = vector_init_from(testing_arena, len, (dahl_fp*)&expect);

    TASK_SUB_VALUE(in, out, 4.0F);

    ASSERT_VECTOR_EQUALS(expect_vec, out);

    // Directly modifies in
    TASK_SUB_VALUE_SELF(in, 4.0F);

    ASSERT_VECTOR_EQUALS(expect_vec, in);

    dahl_arena_reset(testing_arena);
}

void test_matrix_vector_product()
{
    dahl_shape2d constexpr mat_shape = { .x = 3, .y = 2 };
    dahl_fp mat[mat_shape.y][mat_shape.x] = {
        { 1.0F, -1.0F, 2.0F },
        { 0.0F, -3.0F, 1.0F }
    };
    dahl_matrix* in_mat = matrix_init_from(testing_arena, mat_shape, (dahl_fp*)&mat); 

    size_t constexpr in_vec_len = mat_shape.x;
    dahl_fp vec[in_vec_len] = { 2.0F, 1.0F, 0.0F };
    dahl_vector* in_vec = vector_init_from(testing_arena, in_vec_len, (dahl_fp*)&vec);

    size_t constexpr expect_vec_len = mat_shape.y;
    dahl_fp expect[expect_vec_len] = { 1.0F, -3.0F };
    dahl_vector* expect_vec = vector_init_from(testing_arena, expect_vec_len, (dahl_fp*)&expect);

    dahl_vector* out_vec = task_matrix_vector_product_init(testing_arena, in_mat, in_vec);

    ASSERT_VECTOR_EQUALS(expect_vec, out_vec);

    size_t constexpr in_vec_len_2 = mat_shape.y;
    dahl_fp vec_2[in_vec_len_2] = { 2.0F, 4.0F };
    dahl_vector* in_vec_2 = vector_init_from(testing_arena, in_vec_len_2, (dahl_fp*)&vec_2);

    size_t constexpr expect_vec_len_2 = mat_shape.x;
    dahl_fp expect_2[expect_vec_len_2] = { 2.0F, -14.0F, 8.0F };
    dahl_vector* expect_vec_2 = vector_init_from(testing_arena, expect_vec_len_2, (dahl_fp*)&expect_2);

    // Here we need to transpose our matrix
    dahl_matrix* in_mat_t = task_matrix_transpose_init(testing_arena, in_mat);
    dahl_vector* out_vec_2 = task_matrix_vector_product_init(testing_arena, in_mat_t, in_vec_2);

    ASSERT_VECTOR_EQUALS(expect_vec_2, out_vec_2);

    dahl_arena_reset(testing_arena);
}

void test_clip()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* in = vector_init_from(testing_arena, len, (dahl_fp*)&data);

    dahl_vector* out = vector_init(testing_arena, len);

    dahl_fp expect[len] = { 2.0F, 2.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 7.0F, 7.0F };
    dahl_vector* expect_vec = vector_init_from(testing_arena, len, (dahl_fp*)&expect);

    TASK_CLIP(in, out, 2, 7);

    ASSERT_VECTOR_EQUALS(expect_vec, out);

    dahl_fp data_2[len] = { 1e-8F, 1e-8F, 1e-8F, 1e-8F, 1e-8F, 1, 1e-8F, 1e-8F, 1e-8F, 1e-8F };
    dahl_vector* in_2 = vector_init_from(testing_arena, len, (dahl_fp*)&data_2);

    dahl_fp expect_2[len] = { 1e-6F, 1e-6F, 1e-6F, 1e-6F, 1e-6F, 1 - 1e-6F, 1e-6F, 1e-6F, 1e-6F, 1e-6F };
    dahl_vector* expect_vec_2 = vector_init_from(testing_arena, len, (dahl_fp*)&expect_2);

    TASK_CLIP_SELF(in_2, 1e-6F, 1 - 1e-6F);

    ASSERT_VECTOR_EQUALS(expect_vec_2, in_2);

    dahl_arena_reset(testing_arena);
}

void test_cross_entropy_loss()
{
    dahl_shape2d constexpr pred_shape = { .x = 10, .y = 1 }; // 10 classes, 1 batch size
    dahl_fp pred[1][10] = {{ 
        1.69330994e-43F, 1.00000000e+00F, 1.46134680e-11F, 4.19037620e-45F, 2.11622997e-31F, 
        7.47873538e-12F, 5.96985145e-26F, 2.43828226e-41F, 1.16977452e-31F, 1.15460362e-36F
    }};

    dahl_matrix* pred_mat = matrix_init_from(testing_arena, pred_shape, (dahl_fp*)&pred);

    dahl_fp targets[1][10] = {{ 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F }};

    dahl_matrix* target_mat = matrix_init_from(testing_arena, pred_shape, (dahl_fp*)&targets);

    dahl_scalar* res = task_cross_entropy_loss_batch_init(testing_arena, pred_mat, target_mat);
    ASSERT_FP_EQUALS(scalar_get_value(res), 2.461150171736360);
}

void test_matrix_matrix_product()
{
    dahl_shape2d constexpr a_shape = { .x = 3, .y = 2 };
    dahl_fp a[a_shape.y][a_shape.x] = {
        { 1.0F, -1.0F, 2.0F },
        { 0.0F, -3.0F, 1.0F }
    };

    dahl_shape2d constexpr b_shape = { .x = 2, .y = 3 };
    dahl_fp b[b_shape.y][b_shape.x] = {
        { 1.0F,  4.0F },
        { 2.0F, -5.0F },
        { 0.0F, -3.0F }
    };

    dahl_shape2d constexpr expect_shape = { .x = 2, .y = 2 };
    dahl_fp expect[expect_shape.y][expect_shape.x] = {
        {-1.0F,  3.0F },
        {-6.0F, 12.0F }
    };

    dahl_matrix* a_vec = matrix_init_from(testing_arena, a_shape, (dahl_fp*)&a);
    dahl_matrix* b_vec = matrix_init_from(testing_arena, b_shape, (dahl_fp*)&b);
    dahl_matrix* c_vec = matrix_init(testing_arena, expect_shape);
    dahl_matrix* expect_vec = matrix_init_from(testing_arena, expect_shape, (dahl_fp*)&expect);

    task_matrix_matrix_product(a_vec, b_vec, c_vec);

    ASSERT_MATRIX_EQUALS(expect_vec, c_vec);

    dahl_arena_reset(testing_arena);
}

void test_cross_entropy_loss_gradient_batch()
{
    size_t constexpr num_classes = 10;
    size_t constexpr batch_size = 1;
    dahl_fp targets[batch_size][num_classes] = {{ 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F }};
    dahl_fp predictions[num_classes] = {
        9.84501704e-1F, 3.43327192e-6F, 4.29544630e-4F, 3.57638159e-6F, 5.04458589e-9F, 
        3.90385373e-5F, 9.91704419e-3F, 3.92555643e-6F, 3.66346782e-7F, 5.10136218e-3F 
    };
    dahl_fp expect[batch_size][num_classes] = {{ +0.228914562258680, +0.085528577269011, +0.085565029732811, +0.085528589508979, +0.085528284058099, +0.085531622590777, +0.086380691080654, -0.914471380626588, +0.085528314959661, +0.085965709167916, }};

    dahl_shape2d shape = { .x = num_classes, .y = batch_size };
    dahl_matrix* targets_vec = matrix_init_from(testing_arena, shape, (dahl_fp*)&targets);
    dahl_matrix* predictions_vec = matrix_init_from(testing_arena, shape, (dahl_fp*)&predictions);
    dahl_matrix* expect_vec = matrix_init_from(testing_arena, shape, (dahl_fp*)&expect);

    dahl_matrix* gradient = task_cross_entropy_loss_gradient_batch_init(testing_arena, predictions_vec, targets_vec);

    ASSERT_MATRIX_EQUALS_ROUND(expect_vec, gradient, 15);

    dahl_arena_reset(testing_arena);
}

void test_sum()
{
    dahl_block* block = BLOCK(testing_arena, 2, 3, 4, {
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
    });

    dahl_scalar* result = TASK_SUM_INIT(testing_arena, block);

    ASSERT_FP_EQUALS(8, scalar_get_value(result));

    dahl_matrix* matrix = MATRIX(testing_arena, 3, 4, {
        {-2.0F, 1.0F, 2.0F,-1.0F },
        { 3.0F, 1.0F,-3.0F, 1.0F },
        { 4.0F,-1.0F, 4.0F,-1.0F },
    });

    result = TASK_SUM_INIT(testing_arena, matrix);

    ASSERT_FP_EQUALS(8, scalar_get_value(result));

    dahl_vector* vector = VECTOR(testing_arena, 4, { -2.0F, 1.0F, 2.0F,-3.0F });
    result = TASK_SUM_INIT(testing_arena, vector);

    ASSERT_FP_EQUALS(-2, scalar_get_value(result));

    dahl_arena_reset(testing_arena);
}

void assert_convolution_2d(dahl_fp* a, dahl_shape3d a_shape,
                           dahl_fp* b, dahl_shape3d b_shape,
                           dahl_fp* expect, dahl_shape2d expect_shape)
{
    dahl_block* a_matrix = block_init_from(testing_arena, a_shape, a);
    dahl_block* b_matrix = block_init_from(testing_arena, b_shape, b);
    dahl_matrix* c_matrix = matrix_init(testing_arena, expect_shape);
    dahl_matrix* expect_matrix = matrix_init_from(testing_arena, expect_shape, expect);

    task_convolution_2d(a_matrix, b_matrix, c_matrix);

    ASSERT_MATRIX_EQUALS(expect_matrix, c_matrix);

    dahl_arena_reset(testing_arena);
}

void test_convolution_2d_1()
{
    dahl_shape3d a_shape = { .x = 5, .y = 5, .z = 2 };
    dahl_shape3d b_shape = { .x = 3, .y = 3, .z = 2 };
    dahl_shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

    dahl_fp a[2][5][5] = {
        {
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        },
        {
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        }
    };

    dahl_fp b[2][3][3] = {
        {
            { 1.0F, 0.0F, 1.0F },
            { 0.0F, 1.0F, 0.0F },
            { 1.0F, 0.0F, 1.0F },
        },
        {
            { 1.0F, 0.0F, 1.0F },
            { 0.0F, 1.0F, 0.0F },
            { 1.0F, 0.0F, 1.0F },
        }
    }; 

    dahl_fp expect[3][3] = {
        { 10.0F, 10.0F, 10.0F },
        { 10.0F, 10.0F, 10.0F },
        { 10.0F, 10.0F, 10.0F },
    };

    assert_convolution_2d((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);
}

void test_convolution_2d_2()
{
    dahl_shape3d a_shape = { .x = 5, .y = 5, .z = 2 };
    dahl_shape3d b_shape = { .x = 3, .y = 3, .z = 2 };
    dahl_shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

    dahl_fp a[2][5][5] = {
        {
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
            { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        },
        {
            { 2.0F, 2.0F, 2.0F, 2.0F, 2.0F },
            { 2.0F, 2.0F, 2.0F, 2.0F, 2.0F },
            { 2.0F, 2.0F, 2.0F, 2.0F, 2.0F },
            { 2.0F, 2.0F, 2.0F, 2.0F, 2.0F },
            { 2.0F, 2.0F, 2.0F, 2.0F, 2.0F },
        }
    };

    dahl_fp b[2][3][3] = {
        {
            { 1.0F, 0.0F, 1.0F },
            { 0.0F, 1.0F, 0.0F },
            { 1.0F, 0.0F, 1.0F },
        },
        {
            { -1.0F, 0.0F, -1.0F },
            {  0.0F,-1.0F,  0.0F },
            { -1.0F, 0.0F, -1.0F },
        }
    };

    dahl_fp expect[3][3] = {
        { -5.0F, -5.0F, -5.0F },
        { -5.0F, -5.0F, -5.0F },
        { -5.0F, -5.0F, -5.0F },
    };

    assert_convolution_2d((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);
}

void test_convolution_2d_3()
{
    //  This test has been verified with Pytorch Conv2d
    dahl_shape3d a_shape = { .x = 5, .y = 5, .z = 2 };
    dahl_shape3d b_shape = { .x = 3, .y = 3, .z = 2 };
    dahl_shape2d expect_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1 };

    dahl_fp a[2][5][5]={
    {{0.59205108880996704101562500000000,0.78951060771942138671875000000000,0.84097003936767578125000000000000,0.16367709636688232421875000000000,0.75610184669494628906250000000000},{0.64376521110534667968750000000000,0.58918631076812744140625000000000,0.67465502023696899414062500000000,0.91754686832427978515625000000000,0.06646823883056640625000000000000},{0.65309315919876098632812500000000,0.94849878549575805664062500000000,0.13514626026153564453125000000000,0.75979894399642944335937500000000,0.81382769346237182617187500000000},{0.92390638589859008789062500000000,0.44521778821945190429687500000000,0.78127038478851318359375000000000,0.35765880346298217773437500000000,0.65414774417877197265625000000000},{0.03485238552093505859375000000000,0.17248851060867309570312500000000,0.99831753969192504882812500000000,0.91161638498306274414062500000000,0.89625513553619384765625000000000}},
    {{0.90091997385025024414062500000000,0.41513484716415405273437500000000,0.09227979183197021484375000000000,0.80713307857513427734375000000000,0.78480571508407592773437500000000},{0.32392972707748413085937500000000,0.21843320131301879882812500000000,0.04869788885116577148437500000000,0.23623895645141601562500000000000,0.26213979721069335937500000000000},{0.65476948022842407226562500000000,0.19712358713150024414062500000000,0.80608767271041870117187500000000,0.50571846961975097656250000000000,0.07585120201110839843750000000000},{0.79880219697952270507812500000000,0.32912021875381469726562500000000,0.53526717424392700195312500000000,0.18827801942825317382812500000000,0.89535051584243774414062500000000},{0.21847438812255859375000000000000,0.29743838310241699218750000000000,0.09234535694122314453125000000000,0.67999804019927978515625000000000,0.85920387506484985351562500000000}}
    };

    dahl_fp b[2][3][3] = {{{
           0.05701988935470581054687500000000,
           0.88592827320098876953125000000000,
           0.83088457584381103515625000000000},
          {0.93336266279220581054687500000000,
           0.48149150609970092773437500000000,
           0.91633749008178710937500000000000},
          {0.43029594421386718750000000000000,
           0.05245298147201538085937500000000,
           0.50164175033569335937500000000000}},

         {{0.53839641809463500976562500000000,
           0.92247486114501953125000000000000,
           0.37149930000305175781250000000000},
          {0.25306093692779541015625000000000,
           0.74061822891235351562500000000000,
           0.19266653060913085937500000000000},
          {0.24401098489761352539062500000000,
           0.45625072717666625976562500000000,
           0.32857942581176757812500000000000}}
    };

    dahl_fp expect[3][3] = {
        {5.00328372855267744512275385204703,4.76536051742498756311761098913848,4.23578591796455228291051753330976},{4.50411978008361302272533066570759,4.83323681468822030637966236099601,4.27276294436489223471653531305492},{4.90950357166955697607590991538018,4.40020548350297246997797628864646,5.77604361486034534323152911383659}
    };

    assert_convolution_2d((dahl_fp*)&a, a_shape, (dahl_fp*)&b, b_shape, (dahl_fp*)&expect, expect_shape);
}

void test_check_predictions_batch()
{
    // Here we test with 3 classes and 2 batch
    dahl_matrix* pred_batch = MATRIX(testing_arena, 2, 3, {
        { 0.1, 0.2, 0.7 }, // <- This is a wrong prediction
        { 0.0, 0.9, 0.1 }, // <- This is a good prediction
    });

    dahl_matrix* targ_batch = MATRIX(testing_arena, 2, 3, {
        { 0.0, 1.0, 0.0 },
        { 0.0, 1.0, 0.0 },
    });

    dahl_scalar* correct_predictions = task_check_predictions_batch_init(testing_arena, pred_batch, targ_batch);

    ASSERT_FP_EQUALS(1, scalar_get_value(correct_predictions));
}

void test_max_pooling()
{
    dahl_matrix* in = MATRIX(testing_arena, 4, 6, {
        { 1, 2, 3, 8, 9, 3 },
        { 2, 3, 1, 1, 0, 0 },
        { 1, 0, 4, 9, 0, 3 },
        { 8, 2, 2, 7, 2, 0 },
    });

    dahl_shape2d output_shape = { .x = 3, .y = 2 };
    dahl_matrix* output = matrix_init(testing_arena, output_shape);
    dahl_shape2d in_shape = { .x = 6, .y = 4 };
    dahl_matrix* mask = matrix_init(testing_arena, in_shape);

    dahl_matrix* expect = MATRIX(testing_arena, 2, 3, {
        { 3, 8, 9 },
        { 8, 9, 3 },
    });

    task_matrix_max_pooling(in, mask, output, 2);

    ASSERT_MATRIX_EQUALS(expect, output);

    // We verify that the mask is getting populated correctly
    dahl_matrix* expect_mask = MATRIX(testing_arena, 4, 6, {
        { 0, 0, 0, 1, 1, 0 },
        { 0, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 1, 0, 1 },
        { 1, 0, 0, 0, 0, 0 },
    });

    ASSERT_MATRIX_EQUALS(expect_mask, mask);
}

void test_backward_max_pooling()
{
    dahl_matrix* in = MATRIX(testing_arena, 2, 3, {
        { 3, 8, 9 },
        { 8, 9, 3 },
    });

    dahl_matrix* mask = MATRIX(testing_arena, 4, 6, {
        { 0, 0, 0, 1, 1, 0 },
        { 0, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 1, 0, 1 },
        { 1, 0, 0, 0, 0, 0 },
    });

    dahl_shape2d output_shape = { .x = 6, .y = 4 };
    dahl_matrix* output = matrix_init(testing_arena, output_shape);

    dahl_matrix* expect = MATRIX(testing_arena, 4, 6, {
        { 0, 0, 0, 8, 9, 0 },
        { 0, 3, 0, 0, 0, 0 },
        { 0, 0, 0, 9, 0, 3 },
        { 8, 0, 0, 0, 0, 0 },
    });

    task_matrix_backward_max_pooling(in, mask, output, 2);

    ASSERT_MATRIX_EQUALS(expect, output);
}

void test_block_sum_xy_axes()
{
    dahl_block* block = BLOCK(testing_arena, 2, 3, 4, {
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
    });

    dahl_vector* expect = VECTOR(testing_arena, 2, { 8.0F, 0.0F });
    dahl_vector* out = task_block_sum_xy_axes_init(testing_arena, block);
    ASSERT_VECTOR_EQUALS(expect, out);
}

void test_fill()
{
    dahl_vector* vector = VECTOR(testing_arena, 5, { 8.0F, 0.0F, 5.0F, -1.0F, -42.0F });
    dahl_vector* expect = VECTOR(testing_arena, 5, { -667.0F, -667.0F, -667.0F, -667.0F, -667.0F });
    TASK_FILL(vector, -667.0F);
    ASSERT_VECTOR_EQUALS(expect, vector);
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

    dahl_block* block = block_init_from(testing_arena, data_shape, (dahl_fp*)&data);

    dahl_shape3d padded_shape = { .x = 6, .y = 5, .z = 4 };
    dahl_block* padded_block = task_block_add_padding_init(testing_arena, block, padded_shape);

    dahl_block* expect_block = block_init_from(testing_arena, padded_shape, (dahl_fp*)&expect);

    ASSERT_BLOCK_EQUALS(expect_block, padded_block);

    dahl_shape3d padded_shape_2 = { .x = 8, .y = 7, .z = 2 };
    dahl_block* padded_block_2 = task_block_add_padding_init(testing_arena, block, padded_shape_2);

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

    dahl_block* expect_block_2 = block_init_from(testing_arena, padded_shape_2, (dahl_fp*)&expect_2);

    ASSERT_BLOCK_EQUALS(expect_block_2, padded_block_2);

    dahl_arena_reset(testing_arena);
}

void test_matrix_rotate_180()
{
    dahl_matrix* matrix = MATRIX(testing_arena, 3, 4, {
        { 3.0F, 1.0F,-8.0F,-3.0F },
        {-7.0F,-3.0F, 3.0F, 2.0F },
        { 1.0F, 1.0F, 9.0F, 1.0F },
    });

    dahl_matrix* expect = MATRIX(testing_arena, 3, 4, {
        { 1.0F, 9.0F, 1.0F, 1.0F },
        { 2.0F, 3.0F,-3.0F,-7.0F },
        {-3.0F,-8.0F, 1.0F, 3.0F },
    });

    dahl_matrix* out = task_matrix_rotate_180_init(testing_arena, matrix);
    ASSERT_MATRIX_EQUALS(expect, out);

    dahl_arena_reset(testing_arena);
}

void test_vector_outer_product()
{
    dahl_vector* a = VECTOR(testing_arena, 4, { -4, 5, 6, -7 });
    dahl_vector* b = VECTOR(testing_arena, 3, { 1, -2, 3 });

    dahl_matrix* expect = MATRIX(testing_arena, 3, 4, {
        {-4.0F, 5.0F, 6.0F,-7.0F },
        { 8.0F,-10.0F, -12.0F, 14.0F },
        { -12.0F, 15.0F, 18.0F, -21.0F },
    });

    dahl_matrix* out = task_vector_outer_product_init(testing_arena, a, b);

    ASSERT_MATRIX_EQUALS(expect, out);

    dahl_arena_reset(testing_arena);
}

// FIXME: if multiple tests use the random number generator, the rng order gets shifted
// which results in wrong tests :/
// Find a way to start with a new seed/rng each time?
void test_vector_shuffle()
{
    dahl_vector* vec = VECTOR(testing_arena, 7, { -4, 5, 6, -7, 3, -2, 1 });
    dahl_vector* expect = VECTOR(testing_arena, 7, { 3, 1, -7, 5, -2, 6, -4 });

    task_vector_shuffle(vec);
    ASSERT_VECTOR_EQUALS(expect, vec);

    expect = VECTOR(testing_arena, 7, { -4, -2, 1, 6, 5, 3, -7 });

    task_vector_shuffle(vec);
    ASSERT_VECTOR_EQUALS(expect, vec);

    dahl_arena_reset(testing_arena);
}

void test_min_max()
{
    dahl_matrix* m1 = MATRIX(testing_arena, 3, 4, {
        { 4, -5, 2000000000, 3 },
        { 2, -192830, 0, 29 },
        { -2909078, -5, 2000000001, 138 },
    });

    dahl_scalar* min = TASK_MIN_INIT(testing_arena, m1);
    ASSERT_FP_EQUALS(-2909078, scalar_get_value(min));
    dahl_scalar* max = TASK_MAX_INIT(testing_arena, m1);
    ASSERT_FP_EQUALS(2000000001, scalar_get_value(max));

    dahl_arena_reset(testing_arena);
}

void test_convolution_2d_backward_filters()
{
    dahl_block* a = BLOCK(testing_arena, 2, 5, 5, {
        {
            { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
            { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
            { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
            { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
            { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
        },
        {
            { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
            { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
            { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
            { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
            { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
        }
    });

    dahl_matrix* b = MATRIX(testing_arena, 3, 3, {
        { 1.0F, 0.0F, 1.0F },
        { 0.0F, 1.0F, 0.0F },
        { 1.0F, 0.0F, 1.0F },
    }); 

    dahl_block* expect = BLOCK(testing_arena, 2, 3, 3, {
        {
            { 25.0F, 20.0F, 15.0F },
            { 25.0F, 20.0F, 15.0F },
            { 25.0F, 20.0F, 15.0F },
        },
        {
            { 15.0F, 20.0F, 25.0F },
            { 15.0F, 20.0F, 25.0F },
            { 15.0F, 20.0F, 25.0F },
        },
    });

    dahl_shape3d shape = { .x = 3, .y = 3, .z = 2 };
    dahl_block* out = block_init(testing_arena, shape);

    task_convolution_2d_backward_filters(a, b, out);

    ASSERT_BLOCK_EQUALS(expect, out);

    dahl_arena_reset(testing_arena);
}

void test_convolution_2d_backward_input()
{
    // Here we have a padding of 1
    dahl_matrix* a = MATRIX(testing_arena, 4, 4, {
            { 0, 0, 0, 0 },
            { 0, 3, 4, 0 },
            { 0, 3, 4, 0 },
            { 0, 0, 0, 0 },
    });

    dahl_block* b = BLOCK(testing_arena, 3, 2, 2, {
        {
            { 1, 0 },
            { 1, 0 },
        },
        {
            { 0, 1 },
            { 0, 0 },
        },
        {
            { 1, 2 },
            { 0, 1 },
        }
    }); 

    dahl_block* expect = BLOCK(testing_arena, 3, 3, 3, {
        {
            { 3, 4, 0 },
            { 6, 8, 0 },
            { 3, 4, 0 },
        },
        {
            { 0, 3, 4 },
            { 0, 3, 4 },
            { 0, 0, 0 },
        },
        {
            { 3, 10, 8 },
            { 3, 13, 12 },
            { 0, 3, 4 },
        }
    });

    dahl_shape3d expect_shape = { .x = 3, .y = 3, .z = 3 };
    dahl_block* out = block_init(testing_arena, expect_shape);

    task_convolution_2d_backward_input(a, b, out);

    ASSERT_BLOCK_EQUALS(expect, out);

    // Trying the same with the padding free version
    dahl_matrix* a_no_pad = MATRIX(testing_arena, 2, 2, {
            { 3, 4 },
            { 3, 4 },
    });

    out = block_init(testing_arena, expect_shape);
    task_convolution_2d_backward_input_padding_free(a_no_pad, b, out);

    ASSERT_BLOCK_EQUALS(expect, out);

    dahl_arena_reset(testing_arena);
}

void test_round()
{
    dahl_matrix* mat = MATRIX(testing_arena, 3, 3, {
        { 1.5F, 198.9087988F, 989.29831F },
        { 0.3897F, 1.8F, 0.89F },
        { 1.0F, 0.0F, -1.123947F },
    });

    TASK_ROUND_SELF(mat, 4);

    dahl_matrix* expect_mat = MATRIX(testing_arena, 3, 3, {
        { 1.5F, 198.9088F, 989.2983F },
        { 0.3897F, 1.8F, 0.89F },
        { 1.0F, 0.0F, -1.1239F },
    });

    ASSERT_MATRIX_EQUALS_ROUND(expect_mat, mat, 4);

    dahl_scalar* scal = scalar_init_from(testing_arena, 5.98273098);
    TASK_ROUND_SELF(scal, 3);
    dahl_scalar* expect_scal = scalar_init_from(testing_arena, 5.983);

    ASSERT_SCALAR_EQUALS_ROUND(expect_scal, scal, 3);

    dahl_arena_reset(testing_arena);
}

void test_redux_sum()
{
    size_t constexpr nx = 20;
    size_t constexpr ny = 10;

    dahl_matrix* mat = MATRIX(testing_arena, ny, nx, {
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
        { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
        { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
        { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 },
        { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 },
        { 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 },
        { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 },
        { 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    });

    dahl_scalar* redux = scalar_init_redux(testing_arena);

    matrix_partition_along_y(mat);

    for (size_t y = 0; y < ny; y++)
    {
        TASK_SUM(GET_SUB_VECTOR(mat, y), redux);
    }

    matrix_unpartition(mat);

    dahl_scalar* expect = scalar_init_from(testing_arena, 920);
    ASSERT_SCALAR_EQUALS(expect, redux);

    // Of course reusing the same redux variable still accumulates the results
    matrix_partition_along_y(mat);

    for (size_t y = 0; y < ny; y++)
    {
        TASK_SUM(GET_SUB_VECTOR(mat, y), redux);
    }

    matrix_unpartition(mat);

    expect = scalar_init_from(testing_arena, 1840);
    ASSERT_SCALAR_EQUALS(expect, redux);

    dahl_arena_reset(testing_arena);
}

void test_redux_vector_outer_product()
{
    dahl_vector* v1 = VECTOR(testing_arena, 10, {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    });

    dahl_vector* v2 = VECTOR(testing_arena, 10, {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    });

    dahl_vector* v3 = VECTOR(testing_arena, 10, {
        0, 1, 2, 3, 4, 4, 3, 2, 1, 0
    });

    dahl_vector* v4 = VECTOR(testing_arena, 10, {
        4, 3, 2, 1, 0, 0, 1, 2, 3, 4
    });

    dahl_matrix* mat_redux = matrix_init_redux(testing_arena, (dahl_shape2d){ .x = 10, .y = 10 });

    task_vector_outer_product(v1, v2, mat_redux);
    task_vector_outer_product(v3, v4, mat_redux);

    // Compute the expected result
    dahl_matrix* partial_1 = task_vector_outer_product_init(testing_arena, v1, v2);
    dahl_matrix* partial_2 = task_vector_outer_product_init(testing_arena, v3, v4);
    TASK_ADD_SELF(partial_1, partial_2);

    // Compare the reality
    ASSERT_MATRIX_EQUALS(partial_1, mat_redux);

    dahl_arena_reset(testing_arena);
}

void test_redux_convolution_2d_backward_filters()
{
    // Testing partitioning the redux object itself
    size_t constexpr n_samples = 2;
    dahl_tensor* a = TENSOR(testing_arena, n_samples, 2, 5, 5, {
        {
            {
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
            },
            {
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
            }
        },
        {
            {
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
                { 6.0F, 5.0F, 4.0F, 3.0F, 2.0F },
            },
            {
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
                { 2.0F, 3.0F, 4.0F, 5.0F, 6.0F },
            }
        },
    });

    dahl_block* b = BLOCK(testing_arena, n_samples, 3, 3, {
        {
            { 1.0F, 0.0F, 1.0F },
            { 0.0F, 1.0F, 0.0F },
            { 1.0F, 0.0F, 1.0F },
        },
        {
            { 1.0F, 0.0F, 1.0F },
            { 0.0F, 1.0F, 0.0F },
            { 1.0F, 0.0F, 1.0F },
        },
    });

    dahl_tensor* expect_conv = TENSOR(testing_arena, n_samples, 2, 3, 3, {
        {
            {
                { 50.0F, 40.0F, 30.0F },
                { 50.0F, 40.0F, 30.0F },
                { 50.0F, 40.0F, 30.0F },
            },
            {
                { 30.0F, 40.0F, 50.0F },
                { 30.0F, 40.0F, 50.0F },
                { 30.0F, 40.0F, 50.0F },
            },
        },
        {
            {
                { 50.0F, 40.0F, 30.0F },
                { 50.0F, 40.0F, 30.0F },
                { 50.0F, 40.0F, 30.0F },
            },
            {
                { 30.0F, 40.0F, 50.0F },
                { 30.0F, 40.0F, 50.0F },
                { 30.0F, 40.0F, 50.0F },
            },
        },
    });

    // + dimension t because we will compute the same result two times
    dahl_shape4d shape = { .x = 3, .y = 3, .z = 2, .t = n_samples };
    dahl_tensor* out = tensor_init(testing_arena, shape);

    tensor_partition_along_t(a);
    block_partition_along_z(b);
    tensor_partition_along_t_mut(out);

    for (size_t i = 0; i < n_samples; i++)
    {
        dahl_block const* a_block = GET_SUB_BLOCK(a, i);
        dahl_matrix const* b_mat = GET_SUB_MATRIX(b, i);
        dahl_block* out_block_redux = GET_SUB_BLOCK_MUT(out, i);
        block_enable_redux(out_block_redux);

        // Pretend we do mulitple backward, the two functions results should accumulate correctly
        task_convolution_2d_backward_filters(a_block, b_mat, out_block_redux);
        task_convolution_2d_backward_filters(a_block, b_mat, out_block_redux);
    }

    tensor_unpartition(a);
    block_unpartition(b);
    tensor_unpartition(out);

    ASSERT_TENSOR_EQUALS(expect_conv, out);

    dahl_arena_reset(testing_arena);
}

void test_redux_add()
{
    dahl_block* a = BLOCK(testing_arena, 2, 2, 2, {
        {
            { 2, 9 },
            { 2, 7 },
        },
        {
            { 4, 5 },
            { 3, 5 },
        },
    });

    dahl_block* b = BLOCK(testing_arena, 2, 2, 2, {
        {
            { 0, 7 },
            { 9, 6 },
        },
        {
            { -7, 3 },
            { 2, -1 },
        },
    });

    dahl_block* c = BLOCK(testing_arena, 2, 2, 2, {
        {
            { 8, 2 },
            { 3, 3 },
        },
        {
            { -9, 4 },
            { 7, -2 },
        },
    });

    dahl_block* d = BLOCK(testing_arena, 2, 2, 2, {
        {
            { 2, 1 },
            { 0, -1 },
        },
        {
            { 0, 4 },
            { 3, -2 },
        },
    });

    dahl_block* out = block_init(testing_arena, (dahl_shape3d){ .x = 2, .y = 2, .z = 2 });
    dahl_block* out_tmp = block_init(testing_arena, (dahl_shape3d){ .x = 2, .y = 2, .z = 2 });
    TASK_ADD(a, b, out);
    TASK_ADD(c, d, out_tmp);
    TASK_ADD_SELF(out, out_tmp);

    dahl_block* out_self = block_init(testing_arena, (dahl_shape3d){ .x = 2, .y = 2, .z = 2 });
    TASK_ADD_SELF(out_self, a);
    TASK_ADD_SELF(out_self, b);
    TASK_ADD_SELF(out_self, c);
    TASK_ADD_SELF(out_self, d);

    dahl_block* out_redux = block_init_redux(testing_arena, (dahl_shape3d){ .x = 2, .y = 2, .z = 2 });
    TASK_ADD(a, b, out_redux);
    TASK_ADD(c, d, out_redux);

    dahl_block* expect = BLOCK(testing_arena, 2, 2, 2, {
        {
            { 12, 19 },
            { 14, 15 },
        },
        {
            { -12, 16 },
            { 15, 0 },
        },
    });

    ASSERT_BLOCK_EQUALS(expect, out);
    ASSERT_BLOCK_EQUALS(expect, out_self);
    ASSERT_BLOCK_EQUALS(expect, out_redux);

    dahl_arena_reset(testing_arena);
}

void test_redux_sub()
{
    dahl_shape4d shape = { .x = 5, .y = 5, .z = 3, .t = 1 };
    dahl_tensor* a = tensor_init_random(testing_arena, shape, 10, 20);
    dahl_tensor* b = tensor_init_random(testing_arena, shape, 10, 20);
    dahl_tensor* c = tensor_init_random(testing_arena, shape, 10, 20);
    dahl_tensor* d = tensor_init_random(testing_arena, shape, 10, 20);

    // Computing expect without redux mode to verify the result after
    dahl_tensor* expect = tensor_init(testing_arena, shape);
    dahl_tensor* tmp = tensor_init(testing_arena, shape);
    TASK_SUB(a, b, expect);
    TASK_SUB(c, d, tmp);
    TASK_ADD_SELF(expect, tmp);

    dahl_tensor* out = tensor_init_redux(testing_arena, shape);

    TASK_SUB(a, b, out);
    TASK_SUB(c, d, out);

    ASSERT_TENSOR_EQUALS_ROUND(expect, out, 14);

    dahl_arena_reset(testing_arena);
}

void test_vector_matrix_product()
{
    dahl_vector* vector = VECTOR(testing_arena, 3, { 1, 2, 3 });

    dahl_matrix* matrix = MATRIX(testing_arena, 3, 4, {
        { 3.0F, 1.0F,-8.0F,-3.0F },
        {-7.0F,-3.0F, 3.0F, 2.0F },
        { 1.0F, 1.0F, 9.0F, 1.0F },
    });

    dahl_vector* expect = VECTOR(testing_arena, 4, { -8, -2, 25, 4 });
    dahl_vector* out = task_vector_matrix_product_init(testing_arena, vector, matrix);
    ASSERT_VECTOR_EQUALS(expect, out);
}

void test_tasks()
{
    test_tensor_sum_t_axis();
    test_tensor_sum_xyt_axis();
    // test_block_sum_z_axis();
    // test_block_sum_xy_axes();
    // test_block_add_padding();
    // test_matrix_cross_correlation_1();
    // test_matrix_cross_correlation_2();
    // test_matrix_sum_y_axis();
    // test_matrix_vector_product();
    // test_matrix_matrix_product();
    // test_matrix_rotate_180();
    test_relu();
    test_relu_backward();
    test_scal();
    test_power();
    // test_divide();
    test_sub();
    test_add();
    test_add_value();
    // test_vector_softmax();
    // test_vector_dot_product();
    // test_vector_diag();
    // test_vector_outer_product();
    // // test_vector_shuffle();
    // test_vector_matrix_product();
    // test_sub_value();
    // test_clip();
    // test_cross_entropy_loss();
    // test_cross_entropy_loss_gradient_batch();
    // test_sum();
    // test_convolution_2d_1();
    // test_convolution_2d_2();
    // test_convolution_2d_3();
    // test_check_predictions_batch();
    // test_max_pooling();
    // test_backward_max_pooling();
    // test_fill();
    // test_min_max();
    // test_convolution_2d_backward_filters();
    // test_convolution_2d_backward_input();
    // test_round();
    // test_redux_add();
    // test_redux_sub();
    // test_redux_vector_outer_product();
    // test_redux_sum();
    // test_redux_convolution_2d_backward_filters();
}
