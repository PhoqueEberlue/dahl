#include "../include/dahl.h"
#include "starpu_helper.h"
#include "stdlib.h"
#include <stdio.h>

#define LEARNING_RATE 0.001F
#define N_EPOCHS 5

void train_network(dahl_arena* scratch_arena, dahl_arena* network_arena, dahl_dataset* dataset,
                   dahl_convolution* conv, dahl_relu* relu, dahl_pooling* pool, dahl_dense* dense,
                   size_t batch_size, size_t num_samples)
{
    dahl_arena* epoch_arena = dahl_arena_new();
    dahl_arena* batch_arena = dahl_arena_new(); // will be reseted after each batch

    tensor_partition_along_t_batch(dataset->train_images, DAHL_READ, batch_size);
    matrix_partition_along_y_batch(dataset->train_labels, DAHL_READ, batch_size);

    num_samples = 600; // Only use first 1k samples for now
    size_t const n_batches_per_epoch = num_samples / batch_size; // Number of batch we want to do per epoch, not to be confused with batch size

    // Store accuracy and loss here
    dahl_shape2d results_shape = { .x = 2, .y = N_EPOCHS };
    dahl_matrix* results = matrix_init(network_arena, results_shape);

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        dahl_scalar* total_loss = scalar_init(epoch_arena);
        dahl_scalar* correct_predictions = scalar_init(epoch_arena);

        for (size_t i = 0; i < n_batches_per_epoch; i++)
        {
            dahl_tensor const* image_batch = GET_SUB_TENSOR(dataset->train_images, i);
            dahl_matrix const* target_batch = GET_SUB_MATRIX(dataset->train_labels, i);

            dahl_tensor* conv_out = convolution_forward(batch_arena, conv, image_batch);
            relu_forward(relu, conv_out);

            dahl_tensor* pool_out = pooling_forward(batch_arena, pool, conv_out);
            dahl_matrix* pool_out_flattened = tensor_flatten_along_t_no_copy_partition(pool_out);

            dahl_matrix* dense_out = dense_forward(batch_arena, dense, pool_out_flattened); // Returns the predictions for each batch 
            // dahl_scalar* loss = task_cross_entropy_loss_batch_init(batch_arena, dense_out, target_batch);
            // TASK_ADD_SELF(total_loss, loss);

            // dahl_scalar* correct_predictions_batch = task_check_predictions_batch_init(batch_arena, dense_out, target_batch);
            // TASK_ADD_SELF(correct_predictions, correct_predictions_batch);

            dahl_matrix* gradients = task_cross_entropy_loss_gradient_batch_init(batch_arena, dense_out, target_batch); 

            dahl_matrix* dense_back = dense_backward(batch_arena, dense, gradients, pool_out_flattened, LEARNING_RATE);

            dahl_tensor* dense_back_unflattened = matrix_to_tensor_no_copy_partition(dense_back, pool->output_shape);
            dahl_tensor* pool_back = pooling_backward(batch_arena, pool, dense_back_unflattened);
            relu_backward(relu, pool_back);

            dahl_tensor* conv_back = convolution_backward(batch_arena, conv, pool_back, LEARNING_RATE, image_batch);

            dahl_arena_reset(batch_arena);
            dahl_arena_reset(scratch_arena);
            dahl_shutdown();exit(0);
        }
        
        dahl_fp epoch_accuracy = scalar_get_value(correct_predictions) / (dahl_fp)num_samples;
        // the loss already gets divided by batch size so here we divide only by number of batches
        dahl_fp epoch_loss = scalar_get_value(total_loss) / (dahl_fp)n_batches_per_epoch;

        matrix_set_value(results, 0, epoch, epoch_accuracy);
        matrix_set_value(results, 1, epoch, epoch_loss);

        printf("Epoch: %lu, Loss: %f, Accuracy: %f\n",
            epoch,
            epoch_loss, 
            epoch_accuracy
        );

        dahl_arena_reset(epoch_arena);
    }

    matrix_to_csv(results, "dahl-training-outputs.csv", (char const*[2]){"accuracy", "loss"});

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

    // dahl_dataset* dataset = dataset_load_fashion_mnist(network_arena, argv[1], argv[2]);
    // dahl_dataset* dataset = dataset_load_cifar_10(network_arena, argv[1]);
    dahl_dataset* dataset = dataset_load_big_fashion(network_arena, "../datasets/big-fashion/styles.csv", "../datasets/big-fashion/images_270_360/");
    //dahl_dataset* dataset = dataset_load_factice(network_arena, (dahl_shape3d){ .x = 512, .y = 512, .z = 3 }, 640);

    dahl_shape4d images_shape = tensor_get_shape(dataset->train_images);

    // FIXME: support batch size that do not divide the dataset size
    size_t const batch_size = 60;
    size_t const num_samples = images_shape.t;
    size_t const num_channels = images_shape.z;
    size_t const num_filters = 4;
    dahl_shape4d const input_shape = {
        .x = images_shape.x, .y = images_shape.y,
        .z = num_channels, .t = batch_size 
    };
    size_t num_classes = dataset->num_classes;
    size_t const filter_size = 3;
    size_t const pool_size = 2;

    // An arena for temporary results that shouldn't leak to their respective scope
    dahl_arena* scratch_arena = dahl_arena_new();

    dahl_convolution* conv = convolution_init(network_arena, scratch_arena, input_shape, filter_size, num_filters);
    dahl_relu* relu = relu_init(network_arena, conv->output_shape);
    dahl_pooling* pool = pooling_init(network_arena, pool_size, conv->output_shape);

    dahl_shape2d const dense_input_shape = {
        .x = pool->output_shape.x * pool->output_shape.y * pool->output_shape.z,
        .y = batch_size,
    };

    dahl_dense* dense = dense_init(network_arena, scratch_arena, dense_input_shape, num_classes);

    train_network(scratch_arena, network_arena, dataset, conv, relu, pool, dense, batch_size, num_samples);

    dahl_arena_delete(network_arena);
    dahl_arena_delete(scratch_arena);

    dahl_shutdown();
    return 0;
}
