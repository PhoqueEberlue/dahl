#include "tests.h"
#include "../src/tasks.h"
#include <stdio.h>

void assert_cross_correlation_2d(dahl_fp* a, shape2d a_shape,
                                 dahl_fp* b, shape2d b_shape,
                                 dahl_fp* expect, shape2d expect_shape)
{
    dahl_matrix* matrix_a = matrix_init_from(a_shape, a);
    dahl_matrix* matrix_b = matrix_init_from(b_shape, b);
    dahl_matrix* matrix_c = matrix_init(expect_shape);
    dahl_matrix* matrix_expected = matrix_init_from(expect_shape, expect);

    task_cross_correlation_2d(matrix_a, matrix_b, matrix_c);

    starpu_task_wait_for_all();

    assert(matrix_equals(matrix_expected, matrix_c));
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

void test_tasks()
{
    test_cross_correlation_2d_1();
    test_cross_correlation_2d_2();
}
