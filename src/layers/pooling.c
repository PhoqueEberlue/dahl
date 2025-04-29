#include "../../include/dahl_pooling.h"
#include "../../include/dahl_tasks.h"
#include <stdlib.h>

// Not much to init because most of the other fields need to be computed depending on input data.
dahl_pooling* pooling_init(size_t const pool_size, dahl_shape3d const input_shape)
{
    dahl_pooling* pool = malloc(sizeof(dahl_pooling));
    *(size_t*)&pool->pool_size = pool_size;
    *(dahl_shape3d*)&pool->input_shape = input_shape;

    // Image dimensions
    *(size_t*)&pool->output_shape.x = pool->input_shape.x / pool->pool_size;
    *(size_t*)&pool->output_shape.y = pool->input_shape.y / pool->pool_size;
    // Channel dimension
    *(size_t*)&pool->output_shape.z = pool->input_shape.z;

    return pool;
}

dahl_block* pooling_forward(dahl_pooling* pool, dahl_block* input_data)
{
    dahl_block* output_data = block_init(pool->output_shape);
    
    pool->mask = block_init(pool->input_shape);

    block_partition_along_z(input_data);
    block_partition_along_z(output_data);
    block_partition_along_z(pool->mask);

    size_t sub_matrix_nb = block_get_sub_matrix_nb(input_data);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix* sub_input = block_get_sub_matrix(input_data, i);
        dahl_matrix* sub_output = block_get_sub_matrix(output_data, i);
        dahl_matrix* sub_mask = block_get_sub_matrix(pool->mask, i);

        task_matrix_max_pooling(sub_input, sub_output, sub_mask, pool->pool_size);
    }

    block_unpartition(input_data);
    block_unpartition(output_data);
    block_unpartition(pool->mask);

    return output_data;
}

dahl_block* pooling_backward(dahl_pooling* pool, dahl_block* dl_dout)
{
    block_partition_along_z(pool->mask);
    block_partition_along_z(dl_dout);

    size_t sub_matrix_nb = block_get_sub_matrix_nb(pool->mask);

    for (int i = 0; i < sub_matrix_nb; i++)
    {
        dahl_matrix* sub_mask = block_get_sub_matrix(pool->mask, i);
        dahl_matrix* sub_dl_dout = block_get_sub_matrix(dl_dout, i);

        task_matrix_backward_max_pooling_self(sub_dl_dout, sub_mask, pool->pool_size);
    }

    block_unpartition(pool->mask);
    block_unpartition(dl_dout);

    return pool->mask;
}
