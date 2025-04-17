#include "tests.h"
#include <stdio.h>

void test_block_partition()
{

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

    assert(res == 302.0F);

    vector_finalize(vec);
    // Here no need to finalize the block
}



void test_data()
{
    test_block_to_vector();
}
