#include "tests.h"
#include <stdio.h>

void test_miscellaneous()
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

    for (size_t cpu = 0; cpu < 2; cpu++)
    {
        dahl_block* block = block_init_from(testing_arena, data_shape, (dahl_fp*)&data);

        for (size_t i = 0; i < 100; i++)
        {
            // Apparently getting shape and printing doesn't block
            // dahl_shape3d shape = block_get_shape(block);
            // shape3d_print(shape);
            //
            // Initializing here does not represent any problem either
            dahl_block* res = block_init(testing_arena, data_shape);
            //
            // Even partitioning works flawlessly as it is asynchronous
            // block_partition_along_z(block);
            // block_partition_along_z_mut(res);

            // for (size_t i = 0; i < GET_NB_CHILDREN(block); i++)
            // {
            //     dahl_matrix const* mat = GET_SUB_MATRIX(block, i);
            //     dahl_matrix* res_mat = GET_SUB_MATRIX_MUT(res, i);
            //     TASK_WAIT(mat, 50000);
            //     TASK_POWER(mat, res_mat, 2);
            // }

            // block_unpartition(block);
            // block_unpartition(res);
            
            TASK_WAIT(block, 50000);
            TASK_POWER(block, res, 2);
        }
        printf("finished submitting everything for cpu %lu\n", cpu);
    }

    dahl_arena_reset(testing_arena);
}
