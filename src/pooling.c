#include "pooling.h"

// Not much to init because most of the other fields need to be computed depending on input data.
pooling* pooling_init(size_t const pool_size)
{
    pooling* pool = malloc(sizeof(pooling));
    *(size_t*)&pool->pool_size = pool_size;

    return pool;
}

dahl_block* pooling_forward(pooling* const pool, dahl_block const* const input)
{
    pool->input_data = input; // TODO: I mean, input value itself isn't changed? though how do we free the memory?
    pool->input_shape = block_get_shape(input);

    pool->output_shape.x = pool->input_shape.x / pool->pool_size;
    pool->output_shape.y = pool->input_shape.y / pool->pool_size;
    pool->output_shape.z = pool->input_shape.z;

    pool->output_data = block_init(pool->output_shape);
    
    block_partition_along_z(pool->input_data);
    block_partition_along_z(pool->output_data);

    size_t sub_matrix_nb = block_get_sub_matrix_nb(pool->input_data);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix* sub_input = block_get_sub_matrix(pool->input_data, i);
        dahl_matrix* sub_output = block_get_sub_matrix(pool->input_data, i);

        task_matrix_max_pooling(sub_input, sub_output, pool->pool_size);
    }

    block_unpartition(pool->input_data);
    block_unpartition(pool->output_data);

    return pool->output_data;
}

dahl_block* pooling_backward(pooling* const pool, dahl_block const* const dl_dout, double const learning_rate)
{

}
