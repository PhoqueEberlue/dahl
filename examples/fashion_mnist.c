#include "../include/dahl.h"
#include "stdlib.h"
#include <stdio.h>

#define LEARNING_RATE 0.01F
#define N_EPOCHS 30

void train_network(dahl_arena* scratch_arena, dahl_dataset* dataset, 
                   dahl_convolution* conv, dahl_pooling* pool, dahl_dense* dense, 
                   size_t batch_size, size_t num_samples)
{
    dahl_arena* epoch_arena = dahl_arena_new();
    dahl_arena* batch_arena = dahl_arena_new(); // will be reseted after each batch

    tensor_partition_along_t_batch(dataset->train_images, batch_size);
    matrix_partition_along_y_batch(dataset->train_labels, batch_size);

    num_samples = 6000; // Only use first 6k samples for now
    size_t const n_batches_per_epoch = num_samples / batch_size; // Number of batch we want to do per epoch, not to be confused with batch size

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        printf("Epoch %lu\n", epoch);

        dahl_scalar* total_loss = scalar_init(epoch_arena);
        dahl_scalar* correct_predictions = scalar_init(epoch_arena);

        for (size_t i = 0; i < n_batches_per_epoch; i++)
        {
            dahl_tensor const* image_batch = GET_SUB_TENSOR(dataset->train_images, i);
            dahl_matrix const* target_batch = GET_SUB_MATRIX(dataset->train_labels, i);

            dahl_tensor* conv_out = convolution_forward(batch_arena, conv, image_batch);
            dahl_tensor* pool_out = pooling_forward(batch_arena, pool, conv_out);

            dahl_matrix* pool_out_flattened = tensor_flatten_along_t_no_copy(pool_out);
            dahl_matrix* dense_out = dense_forward(batch_arena, dense, pool_out_flattened); // Returns the predictions for each batch 
            
            // TODO: remove copy and replace by starpu temporary data inside the cross entropy task?
            dahl_matrix* dense_out_copy = matrix_init(batch_arena, matrix_get_shape(dense_out));
            TASK_COPY(dense_out, dense_out_copy);
            task_cross_entropy_loss_batch(dense_out_copy, target_batch, total_loss);
            task_check_predictions_batch(dense_out, target_batch, correct_predictions);
            dahl_matrix* gradients = task_cross_entropy_loss_gradient_batch_init(batch_arena, dense_out, target_batch); 

            dahl_matrix* dense_back = dense_backward(batch_arena, dense, gradients, pool_out_flattened, dense_out, LEARNING_RATE);

            dahl_tensor* dense_back_unflattened = matrix_to_tensor_no_copy(dense_back, pool->output_shape);

            dahl_tensor* pool_back = pooling_backward(batch_arena, pool, dense_back_unflattened);
            dahl_tensor* conv_back = convolution_backward(batch_arena, conv, pool_back, LEARNING_RATE, image_batch);
            // Why aren't we using bacward convolution result?
            dahl_arena_reset(scratch_arena);
            dahl_arena_reset(batch_arena);
        }

        printf("Average loss: %f - Accuracy: %f\%\n",
           scalar_get_value(total_loss) / (dahl_fp)num_samples,
           scalar_get_value(correct_predictions) / (dahl_fp)num_samples * 100);

        dahl_arena_reset(epoch_arena);
    }

    tensor_unpartition(dataset->train_images);
    matrix_unpartition(dataset->train_labels);
}

int main(int argc, char **argv)
{
    // Set the seed for reproducible results. Also it seems that StarPU might also set the seed somewhere,
    // because it still works when removing this line.
    srand(42);
    dahl_init();

    // Everything instanciated here will remain allocated till the training finishes.
    // So we put the dataset and the layers containing the trainable parameters (weights & biases).
    dahl_arena* network_arena = dahl_arena_new();

    dahl_dataset* dataset = dataset_load_fashion_mnist(network_arena, argv[1], argv[2]);
    // dahl_dataset* dataset = dataset_load_cifar_10(network_arena, argv[1]);
    dahl_shape4d images_shape = tensor_get_shape(dataset->train_images);

    // FIXME: support batch size that do not divide the dataset size
    size_t const batch_size = 10;
    size_t const num_samples = images_shape.t;
    size_t const num_channels = images_shape.z;
    size_t const num_filters = 32;
    dahl_shape4d const input_shape = { 
        .x = images_shape.x, .y = images_shape.y,
        .z = num_channels, .t = batch_size 
    };
    size_t const num_classes = 10;
    size_t const filter_size = 3;
    size_t const pool_size = 2;

    // An arena for temporary results that shouldn't leak to their respective scope
    dahl_arena* scratch_arena = dahl_arena_new();

    dahl_convolution* conv = convolution_init(network_arena, scratch_arena, input_shape, filter_size, num_filters);
    dahl_pooling* pool = pooling_init(network_arena, pool_size, conv->output_shape);

    dahl_shape2d const dense_input_shape = {
        .x = pool->output_shape.x * pool->output_shape.y * pool->output_shape.z,
        .y = batch_size,
    };

    dahl_dense* dense = dense_init(network_arena, scratch_arena, dense_input_shape, num_classes);

    train_network(scratch_arena, dataset, conv, pool, dense, batch_size, num_samples);

    dahl_arena_delete(network_arena);
    dahl_arena_delete(scratch_arena);

    dahl_shutdown();
    return 0;
}
