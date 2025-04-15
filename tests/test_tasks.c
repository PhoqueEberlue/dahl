#include "tests.h"
#include <stdio.h>

// Asserts that "a cross correlation b = expect"
void assert_matrix_cross_correlation(dahl_fp* a, dahl_shape2d a_shape,
                                     dahl_fp* b, dahl_shape2d b_shape,
                                     dahl_fp* expect, dahl_shape2d expect_shape)
{
    dahl_matrix* a_matrix = matrix_init_from(a_shape, a);
    dahl_matrix* b_matrix = matrix_init_from(b_shape, b);
    dahl_matrix* c_matrix = matrix_init(expect_shape);
    dahl_matrix* expect_matrix = matrix_init_from(expect_shape, expect);

    task_matrix_cross_correlation(a_matrix, b_matrix, c_matrix);

    assert(matrix_equals(expect_matrix, c_matrix));

    matrix_finalize(a_matrix);
    matrix_finalize(b_matrix);
    matrix_finalize(c_matrix);
    matrix_finalize(expect_matrix);
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
}

void test_relu()
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

    TASK_RELU_SELF(a_block);

    assert(block_equals(expect_block, a_block));
    block_finalize(a_block);
    block_finalize(expect_block);
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

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_matrix* expect_matrix = matrix_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_matrix* result_matrix = task_block_sum_z_axis(a_block);

    assert(matrix_equals(expect_matrix, result_matrix));
    block_finalize(a_block);
    matrix_finalize(expect_matrix);
    matrix_finalize(result_matrix);
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

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    TASK_SCAL_SELF(a_block, 2);

    assert(block_equals(expect_block, a_block));
    block_finalize(a_block);
    block_finalize(expect_block);
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

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* b_block = block_init_from(b_shape, (dahl_fp*)&b);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_block* result_block = block_init(a_shape);
    TASK_SUB(a_block, b_block, result_block);

    assert(block_equals(expect_block, result_block)); 

    // here it modifies `a` instead of returning the result
    TASK_SUB_SELF(a_block, b_block);
    assert(block_equals(expect_block, a_block));

    block_finalize(a_block);
    block_finalize(b_block);
    block_finalize(result_block);
    block_finalize(expect_block);
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

    dahl_block* a_block = block_init_from(a_shape, (dahl_fp*)&a);
    dahl_block* b_block = block_init_from(b_shape, (dahl_fp*)&b);
    dahl_block* expect_block = block_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_block* result_block = block_init(a_shape);
    TASK_ADD(a_block, b_block, result_block);

    assert(block_equals(expect_block, result_block)); 

    // here it modifies a instead of returning the result
    TASK_ADD_SELF(a_block, b_block);
    assert(block_equals(expect_block, a_block));

    block_finalize(a_block);
    block_finalize(b_block);
    block_finalize(result_block);
    block_finalize(expect_block);
}

void test_vector_softmax()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F };
    dahl_vector* in = vector_init_from(len, (dahl_fp*)&data);
    dahl_vector* out = vector_init(len);

    dahl_fp expect[len] = { 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F, 0.1F };
    dahl_vector* expect_vec = vector_init_from(len, (dahl_fp*)&expect);

    task_vector_softmax(in, out);

    // Note that values are rounded up in order to compare
    assert(vector_equals(expect_vec, out, true));

    dahl_fp data_2[len] = { 1.8F, 3.8F, 8.7F, 6.9F, 3.9F, 12.9F, 6.0F, 3.7F, 6.1F, 3.2F };
    dahl_vector* in_2 = vector_init_from(len, (dahl_fp*)&data_2);

    dahl_fp expect_2[len] = { 0.000015F, 0.000109F, 0.014701F, 0.002430F, 0.000121F, 
                              0.980384F, 0.000988F, 0.000099F, 0.001092F, 0.000060F };
    dahl_vector* expect_vec_2 = vector_init_from(len, (dahl_fp*)&expect_2);

    task_vector_softmax(in_2, out);

    assert(vector_equals(expect_vec_2, out, true));

    vector_finalize(in);
    vector_finalize(out);
    vector_finalize(expect_vec);
    vector_finalize(in_2);
    vector_finalize(expect_vec_2);
}

void test_vector_dot_product()
{
    size_t constexpr len = 10;
    dahl_fp data_1[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* a = vector_init_from(len, (dahl_fp*)&data_1);

    dahl_fp expect = 285.0F;

    dahl_fp result = task_vector_dot_product(a, a);

    assert(expect == result);

    dahl_fp data_2[len] = { 9.0F, 8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.0F };
    dahl_vector* b = vector_init_from(len, (dahl_fp*)&data_2);

    dahl_fp expect_2 = 120.0F;

    result = task_vector_dot_product(a, b);

    assert(expect_2 == result);

    vector_finalize(a);
    vector_finalize(b);
}

void test_vector_diag()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* a = vector_init_from(len, (dahl_fp*)&data);

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
    dahl_matrix* expect_matrix = matrix_init_from(expect_shape, (dahl_fp*)&expect);

    dahl_matrix* result = task_vector_diag(a);

    assert(matrix_equals(expect_matrix, result));

    vector_finalize(a);
    matrix_finalize(expect_matrix);
    matrix_finalize(result);
}

void test_add_value()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* in = vector_init_from(len, (dahl_fp*)&data);

    dahl_vector* out = vector_init(len);

    dahl_fp expect[len] = { 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F };
    dahl_vector* expect_vec = vector_init_from(len, (dahl_fp*)&expect);

    TASK_ADD_VALUE(in, out, 4.0F);

    assert(vector_equals(expect_vec, out, false));

    // Directly modifies in
    TASK_ADD_VALUE_SELF(in, 4.0F);

    assert(vector_equals(expect_vec, in, false));

    vector_finalize(in);
    vector_finalize(out);
    vector_finalize(expect_vec);
}

void test_sub_value()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* in = vector_init_from(len, (dahl_fp*)&data);

    dahl_vector* out = vector_init(len);

    dahl_fp expect[len] = { -4.0F, -3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F };
    dahl_vector* expect_vec = vector_init_from(len, (dahl_fp*)&expect);

    TASK_SUB_VALUE(in, out, 4.0F);

    assert(vector_equals(expect_vec, out, false));

    // Directly modifies in
    TASK_SUB_VALUE_SELF(in, 4.0F);

    assert(vector_equals(expect_vec, in, false));

    vector_finalize(in);
    vector_finalize(out);
    vector_finalize(expect_vec);
}

void test_matrix_vector_product()
{
    dahl_shape2d constexpr mat_shape = { .x = 3, .y = 2 };
    dahl_fp mat[mat_shape.y][mat_shape.x] = {
        { 1.0F, -1.0F, 2.0F },
        { 0.0F, -3.0F, 1.0F }
    };
    dahl_matrix* in_mat = matrix_init_from(mat_shape, (dahl_fp*)&mat); 

    size_t constexpr in_vec_len = mat_shape.x;
    dahl_fp vec[in_vec_len] = { 2.0F, 1.0F, 0.0F };
    dahl_vector* in_vec = vector_init_from(in_vec_len, (dahl_fp*)&vec);

    size_t constexpr expect_vec_len = mat_shape.y;
    dahl_fp expect[expect_vec_len] = { 1.0F, -3.0F };
    dahl_vector* expect_vec = vector_init_from(expect_vec_len, (dahl_fp*)&expect);

    dahl_vector* out_vec = task_matrix_vector_product(in_mat, in_vec);

    assert(vector_equals(expect_vec, out_vec, false));

    matrix_finalize(in_mat);
    vector_finalize(in_vec);
    vector_finalize(out_vec);
    vector_finalize(expect_vec);
}

void test_clip()
{
    size_t constexpr len = 10;
    dahl_fp data[len] = { 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    dahl_vector* in = vector_init_from(len, (dahl_fp*)&data);

    dahl_vector* out = vector_init(len);

    dahl_fp expect[len] = { 2.0F, 2.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 7.0F, 7.0F };
    dahl_vector* expect_vec = vector_init_from(len, (dahl_fp*)&expect);

    TASK_CLIP(in, out, 2, 7);

    assert(vector_equals(expect_vec, out, false));

    dahl_fp data_2[len] = { 1e-8F, 1e-8F, 1e-8F, 1e-8F, 1e-8F, 1, 1e-8F, 1e-8F, 1e-8F, 1e-8F };
    dahl_vector* in_2 = vector_init_from(len, (dahl_fp*)&data_2);

    dahl_fp expect_2[len] = { 1e-6F, 1e-6F, 1e-6F, 1e-6F, 1e-6F, 1 - 1e-6F, 1e-6F, 1e-6F, 1e-6F, 1e-6F };
    dahl_vector* expect_vec_2 = vector_init_from(len, (dahl_fp*)&expect_2);

    TASK_CLIP_SELF(in_2, 1e-6F, 1 - 1e-6F);

    assert(vector_equals(expect_vec_2, in_2, false));

    vector_finalize(in);
    vector_finalize(out);
    vector_finalize(expect_vec);
    vector_finalize(in_2);
    vector_finalize(expect_vec_2);
}

void test_tasks()
{
    test_matrix_cross_correlation_1();
    test_matrix_cross_correlation_2();
    test_relu();
    test_block_sum_z_axis();
    test_scal();
    test_sub();
    test_add();
    test_vector_softmax();
    test_vector_dot_product();
    test_vector_diag();
    test_add_value();
    test_sub_value();
    test_matrix_vector_product();
    test_clip();
}
