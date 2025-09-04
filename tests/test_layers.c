#include "layers_data.h"
#include "../include/dahl_convolution.h"
#include "tests.h"
#include <stdio.h>

void test_convolution()
{
    dahl_arena* scratch_arena = dahl_arena_new();

    // ----------- Forward -----------
    dahl_convolution* conv = convolution_init(testing_arena, scratch_arena, conv_input_shape, filter_size, num_filters);

    tensor_set_from(conv->filters, (dahl_fp*)&init_conv_filters);
    vector_set_from(conv->biases, (dahl_fp*)&init_conv_biases);

    dahl_tensor const* img_batch = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&init_sample);

    dahl_tensor* conv_forward_out = convolution_forward(testing_arena, conv, img_batch);

    dahl_tensor const* expect_forward = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_conv_forward);

    ASSERT_SHAPE4D_EQUALS(conv_output_shape, tensor_get_shape(conv_forward_out));
    ASSERT_TENSOR_EQUALS_ROUND(expect_forward, conv_forward_out, 12);

    // ----------- Backward -----------
    dahl_tensor const* pool_backward = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_pool_backward);

    dahl_tensor* conv_backward_out = convolution_backward(testing_arena, conv, pool_backward, learning_rate, img_batch);

    dahl_tensor const* expect_backward = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&expect_conv_backward);

    ASSERT_SHAPE4D_EQUALS(conv_input_shape, tensor_get_shape(conv_backward_out));
    // TODO: thought It seems I cannot access those data with pytorch
    // ASSERT_TENSOR_EQUALS_ROUND(expect_backward, conv_backward_out, 6);

    // testing that weights and biases are correctly updated
    dahl_shape4d const expect_filters_shape = { .x = filter_size, .y = filter_size, .z = img_nz, .t = num_filters };
    dahl_tensor const* expect_filters = tensor_init_from(testing_arena, expect_filters_shape, (dahl_fp*)&expect_conv_filters);
    dahl_vector const* expect_biases = vector_init_from(testing_arena, num_filters, (dahl_fp*)&expect_conv_biases);

    ASSERT_SHAPE4D_EQUALS(expect_filters_shape, tensor_get_shape(conv->filters));
    ASSERT_TENSOR_EQUALS_ROUND(expect_filters, conv->filters, 3);

    ASSERT_SIZE_T_EQUALS(num_filters, vector_get_len(conv->biases));
    ASSERT_VECTOR_EQUALS_ROUND(expect_biases, conv->biases, 4);

    dahl_arena_reset(testing_arena);
    dahl_arena_delete(scratch_arena);
}

void test_pool()
{
    // ----------- Forward -----------
    dahl_pooling* pool = pooling_init(testing_arena, pool_size, conv_output_shape);
    dahl_tensor* input = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_conv_forward);

    dahl_tensor* pool_forward_out = pooling_forward(testing_arena, pool, input);

    dahl_tensor const* expect_forward = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_pool_forward);

    ASSERT_SHAPE4D_EQUALS(pool_output_shape, tensor_get_shape(pool_forward_out));
    ASSERT_TENSOR_EQUALS(expect_forward, pool_forward_out);

    dahl_shape4d constexpr expect_mask_shape = { .x = 26, .y = 26, .z = num_filters, .t = batch_size };
    dahl_tensor const* expect_mask = tensor_init_from(testing_arena, expect_mask_shape, (dahl_fp*)&expect_pool_mask);
    ASSERT_TENSOR_EQUALS(expect_mask, pool->mask_batch);

    // ----------- Backward -----------
    dahl_tensor* input_backward = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_dense_backward);

    dahl_tensor* pool_backward_out = pooling_backward(testing_arena, pool, input_backward); 

    dahl_tensor const* expect_backward = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_pool_backward);

    ASSERT_SHAPE4D_EQUALS(conv_output_shape, tensor_get_shape(pool_backward_out));
    ASSERT_TENSOR_EQUALS(expect_backward, pool_backward_out);

    // No weights/biases on this layer

    dahl_arena_reset(testing_arena);
}

void test_dense()
{
    dahl_arena* scratch_arena = dahl_arena_new();

    // ----------- Forward -----------
    // Start to flatten pooling output
    dahl_tensor const* input = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_pool_forward);
    dahl_matrix const* input_flattened = tensor_flatten_along_t_no_copy(input);

    ASSERT_SHAPE2D_EQUALS(dense_input_shape, matrix_get_shape(input_flattened));

    dahl_dense* dense = dense_init(testing_arena, scratch_arena, dense_input_shape, num_classes);

    matrix_set_from(dense->weights, (dahl_fp*)&init_dense_weights);
    vector_set_from(dense->biases, init_dense_biases);

    dahl_matrix* dense_forward_out = dense_forward(testing_arena, dense, input_flattened);

    dahl_matrix const* expect_forward = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_forward);

    ASSERT_MATRIX_EQUALS_ROUND(expect_forward, dense_forward_out, 13);

    // ----------- Backward ----------- 
    dahl_matrix const* targets = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&init_targets);

    dahl_scalar* loss = task_cross_entropy_loss_batch_init(testing_arena, expect_forward, targets);

    ASSERT_FP_EQUALS_ROUND(expect_dense_loss, scalar_get_value(loss), 14);

    dahl_matrix* gradient_batch = task_cross_entropy_loss_gradient_batch_init(testing_arena, expect_forward, targets);
    dahl_matrix const* expect_gradients = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_gradients);

    ASSERT_MATRIX_EQUALS_ROUND(expect_gradients, gradient_batch, 13);

    dahl_matrix* dense_backward_out = dense_backward(testing_arena, dense, expect_gradients, input_flattened, expect_forward, learning_rate);

    dahl_matrix const* expect_backward = matrix_init_from(testing_arena, dense_input_shape, (dahl_fp*)&expect_dense_backward);

    ASSERT_SHAPE2D_EQUALS(dense_input_shape, matrix_get_shape(dense_backward_out));
    ASSERT_MATRIX_EQUALS_ROUND(expect_backward, dense_backward_out, 11);

    // testing that weights and biases are correctly updated
    dahl_shape2d const expect_weigths_shape = { .x = 676, .y = num_classes };
    dahl_matrix const* expect_weights = matrix_init_from(testing_arena, expect_weigths_shape, (dahl_fp*)&expect_dense_weights);
    dahl_vector const* expect_biases = vector_init_from(testing_arena, num_classes, (dahl_fp*)&expect_dense_biases);

    ASSERT_SHAPE2D_EQUALS(expect_weigths_shape, matrix_get_shape(dense->weights));
    // FIXME: big drop in precision
    ASSERT_MATRIX_EQUALS_ROUND(expect_weights, dense->weights, 3);
    ASSERT_VECTOR_EQUALS_ROUND(expect_biases, dense->biases, 4);

    dahl_arena_reset(testing_arena);
    dahl_arena_delete(scratch_arena);
}

void test_layers()
{
    test_convolution();
    test_pool();
    test_dense();   
}
