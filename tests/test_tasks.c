#include "tests.h"
#include "../src/tasks.h"
#include "../src/utils.h"

void test_cross_correlation_2d()
{
    shape3d a_shape = { .x = 5, .y = 5, .z = 1 };
    dahl_fp a[1][5][5] = { {
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
        { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F },
    },};

    starpu_data_handle_t a_handle = block_init_from(a_shape, (dahl_fp*) &a);

    shape3d b_shape = { .x = 3, .y = 3, .z = 1 };
    dahl_fp b[1][3][3] = { {
        { 1.0F, 0.0F, 1.0F },
        { 0.0F, 1.0F, 0.0F },
        { 1.0F, 0.0F, 1.0F },
    },};

    starpu_data_handle_t b_handle = block_init_from(b_shape, (dahl_fp*) &b);

    block_print_from_handle(a_handle);
    block_print_from_handle(b_handle);

    shape3d c_shape = { .x = a_shape.x - b_shape.x + 1, .y = a_shape.y - b_shape.y + 1, .z = a_shape.z - b_shape.z + 1 };
    starpu_data_handle_t c_handle = block_init(c_shape);

    task_cross_correlation_2d(a_handle, b_handle, c_handle);

    block_print_from_handle(c_handle);

    starpu_task_wait_for_all();

    dahl_fp expect[1][3][3] = { {
        { 5.0F, 5.0F, 5.0F },
        { 5.0F, 5.0F, 5.0F },
        { 5.0F, 5.0F, 5.0F },
    },};

    starpu_data_handle_t expect_handle = block_init_from(b_shape, (dahl_fp*) &expect);

    assert(block_equals(expect_handle, c_handle));
}

void test_tasks()
{
    test_cross_correlation_2d();
}
