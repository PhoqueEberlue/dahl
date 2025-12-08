#include "test_layers_data.h"
#include "tests.h"
#include <stdio.h>

void test_convolution()
{
    dahl_arena* scratch_arena = dahl_arena_new();

    // ----------- Forward -----------
    dahl_convolution* conv = convolution_init(testing_arena, scratch_arena, conv_input_shape, filter_size, num_filters);
    dahl_relu* relu = relu_init(testing_arena, conv->output_shape);

    tensor_set_from(conv->filters, (dahl_fp*)&init_conv_filters);
    vector_set_from(conv->biases, (dahl_fp*)&init_conv_biases);

    dahl_tensor* img_batch = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&init_sample);
    tensor_partition_along_t(img_batch, DAHL_READ);

    dahl_tensor_part* conv_forward_out = convolution_forward(testing_arena, conv, img_batch);

    dahl_tensor const* expect_forward = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_conv_forward);

    ASSERT_SHAPE4D_EQUALS(conv_output_shape, tensor_get_shape(conv_forward_out));
    ASSERT_TENSOR_EQUALS_ROUND(expect_forward, conv_forward_out, 11);

    dahl_tensor const* expect_relu = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_relu_forward);

    relu_forward(relu, conv_forward_out);

    ASSERT_TENSOR_EQUALS_ROUND(expect_relu, conv_forward_out, 12);

    // ----------- Backward -----------
    dahl_tensor_part* pool_backward = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_pool_backward);
    tensor_partition_along_t(pool_backward, DAHL_MUT);

    relu_backward(relu, pool_backward);

    dahl_tensor_part* conv_backward_out = convolution_backward(testing_arena, conv, pool_backward, learning_rate, img_batch);

    dahl_tensor const* expect_backward = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&expect_conv_backward);

    ASSERT_SHAPE4D_EQUALS(conv_input_shape, tensor_get_shape(conv_backward_out));
    // TODO: thought It seems I cannot access those data with pytorch
    // ASSERT_TENSOR_EQUALS_ROUND(expect_backward, conv_backward_out, 6);

    // testing that weights and biases are correctly updated
    dahl_shape4d const expect_filters_shape = { .x = filter_size, .y = filter_size, .z = img_nz, .t = num_filters };
    dahl_tensor const* expect_filters = tensor_init_from(testing_arena, expect_filters_shape, (dahl_fp*)&expect_conv_filters);
    dahl_vector const* expect_biases = vector_init_from(testing_arena, num_filters, (dahl_fp*)&expect_conv_biases);

    ASSERT_SHAPE4D_EQUALS(expect_filters_shape, tensor_get_shape(conv->filters));
    ASSERT_TENSOR_EQUALS_ROUND(expect_filters, conv->filters, 14);

    ASSERT_SIZE_T_EQUALS(num_filters, vector_get_len(conv->biases));
    ASSERT_VECTOR_EQUALS_ROUND(expect_biases, conv->biases, 14);

    dahl_arena_reset(testing_arena);
    dahl_arena_delete(scratch_arena);
}

void test_pool()
{
    // ----------- Forward -----------
    dahl_pooling* pool = pooling_init(testing_arena, pool_size, conv_output_shape);

    dahl_tensor* input = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_relu_forward);
    tensor_partition_along_t(input, DAHL_READ);

    dahl_tensor_part* pool_forward_out = pooling_forward(testing_arena, pool, input);

    dahl_tensor const* expect_forward = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_pool_forward);

    ASSERT_SHAPE4D_EQUALS(pool_output_shape, tensor_get_shape(pool_forward_out));
    ASSERT_TENSOR_EQUALS(expect_forward, pool_forward_out);

    dahl_shape4d constexpr expect_mask_shape = { .x = 26, .y = 26, .z = num_filters, .t = batch_size };
    dahl_tensor const* expect_mask = tensor_init_from(testing_arena, expect_mask_shape, (dahl_fp*)&expect_pool_mask);
    ASSERT_TENSOR_EQUALS(expect_mask, pool->mask_batch);

    // ----------- Backward -----------
    dahl_tensor_part* input_backward = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_dense_backward);
    tensor_partition_along_t(input_backward, DAHL_READ);

    dahl_tensor_part* pool_backward_out = pooling_backward(testing_arena, pool, input_backward); 

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
    dahl_tensor* input = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_pool_forward);
    tensor_partition_along_t(input, DAHL_READ);

    // Start to flatten pooling output
    dahl_matrix_part const* input_flattened = tensor_flatten_along_t_no_copy_partition(input);

    dahl_dense* dense = dense_init(testing_arena, scratch_arena, dense_input_shape, num_classes);

    matrix_set_from(dense->weights, (dahl_fp*)&init_dense_weights);
    vector_set_from(dense->biases, init_dense_biases);

    dahl_matrix_part* dense_forward_out = dense_forward(testing_arena, dense, input_flattened);

    dahl_matrix const* expect_forward = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_forward);

    ASSERT_MATRIX_EQUALS_ROUND(expect_forward, dense_forward_out, 13);

    // ----------- Backward ----------- 
    dahl_matrix const* targets = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&init_targets);

    dahl_scalar* loss = task_cross_entropy_loss_batch_init(testing_arena, expect_forward, targets);

    ASSERT_FP_EQUALS_ROUND(expect_dense_loss[0], scalar_get_value(loss), 15);

    matrix_partition_along_y(expect_forward, DAHL_READ);
    matrix_partition_along_y(targets, DAHL_READ);

    dahl_matrix_part* gradient_batch = task_cross_entropy_loss_gradient_batch_init(testing_arena, expect_forward, targets);

    dahl_matrix const* expect_gradients = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_gradients);

    ASSERT_MATRIX_EQUALS_ROUND(expect_gradients, gradient_batch, 14);

    matrix_partition_along_y(expect_gradients, DAHL_READ);

    dahl_matrix_part* dense_backward_out = dense_backward(testing_arena, dense, expect_gradients, input_flattened, learning_rate);

    dahl_matrix const* expect_backward = matrix_init_from(testing_arena, dense_input_shape, (dahl_fp*)&expect_dense_backward);

    ASSERT_SHAPE2D_EQUALS(dense_input_shape, matrix_get_shape(dense_backward_out));
    ASSERT_MATRIX_EQUALS_ROUND(expect_backward, dense_backward_out, 13);

    // testing that weights and biases are correctly updated
    dahl_shape2d const expect_weigths_shape = { .x = 676, .y = num_classes };
    dahl_matrix const* expect_weights = matrix_init_from(testing_arena, expect_weigths_shape, (dahl_fp*)&expect_dense_weights);
    dahl_vector const* expect_biases = vector_init_from(testing_arena, num_classes, (dahl_fp*)&expect_dense_biases);

    ASSERT_SHAPE2D_EQUALS(expect_weigths_shape, matrix_get_shape(dense->weights));
    // Weights see a slight drop in precision, but it is acceptable.
    ASSERT_MATRIX_EQUALS_ROUND(expect_weights, dense->weights, 10);
    ASSERT_VECTOR_EQUALS_ROUND(expect_biases, dense->biases, 15);

    dahl_arena_reset(testing_arena);
    dahl_arena_delete(scratch_arena);
}

// This test actual reuses the data computed by each layer to test the flow of the network.
void test_flow()
{
    dahl_arena* scratch_arena = dahl_arena_new();
    size_t constexpr number_batches = 2;

    // ----------- Initializing layers -----------
    dahl_convolution* conv = convolution_init(testing_arena, scratch_arena, conv_input_shape, filter_size, num_filters);
    tensor_set_from(conv->filters, (dahl_fp*)&init_conv_filters);
    vector_set_from(conv->biases, (dahl_fp*)&init_conv_biases);
    dahl_relu* relu = relu_init(testing_arena, conv->output_shape);

    dahl_pooling* pool = pooling_init(testing_arena, pool_size, conv_output_shape);

    dahl_dense* dense = dense_init(testing_arena, scratch_arena, dense_input_shape, num_classes);
    matrix_set_from(dense->weights, (dahl_fp*)&init_dense_weights);
    vector_set_from(dense->biases, init_dense_biases);

    // ----------- Initializing testing values -----------
    dahl_tensor const* img_batches[number_batches]; 
    dahl_matrix const* target_batches[number_batches]; 
    dahl_tensor const* expect_conv_forward_batch[number_batches]; 
    dahl_tensor const* expect_relu_forward_batch[number_batches]; 
    dahl_tensor const* expect_pool_forward_batch[number_batches]; 
    dahl_tensor const* expect_pool_mask_batch[number_batches]; 
    dahl_matrix const* expect_dense_forward_batch[number_batches]; 
    dahl_matrix const* expect_dense_gradient_batch[number_batches]; 
    dahl_matrix const* expect_dense_backward_batch[number_batches]; 
    dahl_matrix const* expect_dense_weights_batch[number_batches]; 
    dahl_vector const* expect_dense_biases_batch[number_batches]; 
    dahl_tensor const* expect_pool_backward_batch[number_batches]; 
    // dahl_tensor const* expect_conv_backward_batch[number_batches]; 
    dahl_tensor const* expect_conv_weights_batch[number_batches]; 
    dahl_vector const* expect_conv_biases_batch[number_batches]; 

    dahl_shape4d constexpr expect_mask_shape = { .x = 26, .y = 26, .z = num_filters, .t = batch_size };
    dahl_shape2d const expect_weigths_shape = { .x = 676, .y = num_classes };
    dahl_shape4d const expect_filters_shape = { .x = filter_size, .y = filter_size, .z = img_nz, .t = num_filters };

    img_batches[0] = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&init_sample);
    target_batches[0] = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&init_targets);
    expect_conv_forward_batch[0] = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_conv_forward);
    expect_relu_forward_batch[0] = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_relu_forward);
    expect_pool_forward_batch[0] = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_pool_forward);
    expect_pool_mask_batch[0] = tensor_init_from(testing_arena, expect_mask_shape, (dahl_fp*)&expect_pool_mask);
    expect_dense_forward_batch[0] = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_forward);
    expect_dense_gradient_batch[0] = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_gradients);
    expect_dense_backward_batch[0] = matrix_init_from(testing_arena, dense_input_shape, (dahl_fp*)&expect_dense_backward);
    expect_dense_weights_batch[0] = matrix_init_from(testing_arena, expect_weigths_shape, (dahl_fp*)&expect_dense_weights);
    expect_dense_biases_batch[0] = vector_init_from(testing_arena, num_classes, (dahl_fp*)&expect_dense_biases);
    expect_pool_backward_batch[0] = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_pool_backward);
    // expect_conv_backward_batch[0] = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&expect_conv_backward);
    expect_conv_weights_batch[0] = tensor_init_from(testing_arena, expect_filters_shape, (dahl_fp*)&expect_conv_filters);
    expect_conv_biases_batch[0] = vector_init_from(testing_arena, num_filters, (dahl_fp*)&expect_conv_biases);

    img_batches[1] = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&init_sample_1);
    target_batches[1] = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&init_targets_1);
    expect_conv_forward_batch[1] = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_conv_forward_1);
    expect_relu_forward_batch[1] = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_relu_forward_1);
    expect_pool_forward_batch[1] = tensor_init_from(testing_arena, pool_output_shape, (dahl_fp*)&expect_pool_forward_1);
    expect_pool_mask_batch[1] = tensor_init_from(testing_arena, expect_mask_shape, (dahl_fp*)&expect_pool_mask_1);
    expect_dense_forward_batch[1] = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_forward_1);
    expect_dense_gradient_batch[1] = matrix_init_from(testing_arena, dense_output_shape, (dahl_fp*)&expect_dense_gradients_1);
    expect_dense_backward_batch[1] = matrix_init_from(testing_arena, dense_input_shape, (dahl_fp*)&expect_dense_backward_1);
    expect_dense_weights_batch[1] = matrix_init_from(testing_arena, expect_weigths_shape, (dahl_fp*)&expect_dense_weights_1);
    expect_dense_biases_batch[1] = vector_init_from(testing_arena, num_classes, (dahl_fp*)&expect_dense_biases_1);
    expect_pool_backward_batch[1] = tensor_init_from(testing_arena, conv_output_shape, (dahl_fp*)&expect_pool_backward_1);
    // expect_conv_backward_batch[1] = tensor_init_from(testing_arena, conv_input_shape, (dahl_fp*)&expect_conv_backward);
    expect_conv_weights_batch[1] = tensor_init_from(testing_arena, expect_filters_shape, (dahl_fp*)&expect_conv_filters_1);
    expect_conv_biases_batch[1] = vector_init_from(testing_arena, num_filters, (dahl_fp*)&expect_conv_biases_1);

    // ----------- Simulating an epoch --------- 
    for (size_t i = 0; i < number_batches; i++)
    {
        tensor_partition_along_t(img_batches[i],DAHL_READ);
        matrix_partition_along_y(target_batches[i],DAHL_READ);

        // Forward pass
        dahl_tensor_part* conv_forward_out = convolution_forward(testing_arena, conv, img_batches[i]);
        ASSERT_TENSOR_EQUALS_ROUND(expect_conv_forward_batch[i], conv_forward_out, 11);

        relu_forward(relu, conv_forward_out);
        ASSERT_TENSOR_EQUALS_ROUND(expect_relu_forward_batch[i], conv_forward_out, 12);

        dahl_tensor_part* pool_forward_out = pooling_forward(testing_arena, pool, conv_forward_out);
        ASSERT_TENSOR_EQUALS_ROUND(expect_pool_forward_batch[i], pool_forward_out, 12);
        ASSERT_TENSOR_EQUALS(expect_pool_mask_batch[i], pool->mask_batch);

        dahl_matrix_part const* pool_flattened = tensor_flatten_along_t_no_copy_partition(pool_forward_out);
        dahl_matrix_part* dense_forward_out = dense_forward(testing_arena, dense, pool_flattened);
        ASSERT_MATRIX_EQUALS_ROUND(expect_dense_forward_batch[i], dense_forward_out, 13);

        // Backward pass
        dahl_scalar* loss = task_cross_entropy_loss_batch_init(testing_arena, dense_forward_out, target_batches[i]);
        ASSERT_FP_EQUALS_ROUND(expect_dense_loss[i], scalar_get_value(loss), 14);

        dahl_matrix_part* gradient_batch = task_cross_entropy_loss_gradient_batch_init(testing_arena, dense_forward_out, target_batches[i]);
        ASSERT_MATRIX_EQUALS_ROUND(expect_dense_gradient_batch[i], gradient_batch, 14);

        dahl_matrix_part* dense_backward_out = dense_backward(testing_arena, dense, gradient_batch, pool_flattened, learning_rate);
        ASSERT_MATRIX_EQUALS_ROUND(expect_dense_backward_batch[i], dense_backward_out, 12);

        // Weights see a slight drop in precision, but it is acceptable.
        ASSERT_MATRIX_EQUALS_ROUND(expect_dense_weights_batch[i], dense->weights, 10);
        ASSERT_VECTOR_EQUALS_ROUND(expect_dense_biases_batch[i], dense->biases, 14);

        dahl_tensor_part* dense_back_unflattened = matrix_to_tensor_no_copy_partition(dense_backward_out, pool->output_shape);
        dahl_tensor_part* pool_backward_out = pooling_backward(testing_arena, pool, dense_back_unflattened); 
        ASSERT_TENSOR_EQUALS_ROUND(expect_pool_backward_batch[i], pool_backward_out, 12);

        relu_backward(relu, pool_backward_out);

        dahl_tensor_part* conv_backward_out_p = convolution_backward(testing_arena, conv, pool_backward_out, learning_rate, img_batches[i]); 
        ASSERT_TENSOR_EQUALS_ROUND(expect_conv_weights_batch[i], conv->filters, 12);
        ASSERT_VECTOR_EQUALS_ROUND(expect_conv_biases_batch[i], conv->biases, 14);
    }

    dahl_arena_delete(scratch_arena);
    dahl_arena_reset(testing_arena);
}

void test_layers()
{
    test_convolution();
    test_pool();
    test_dense();
    test_flow();
}
