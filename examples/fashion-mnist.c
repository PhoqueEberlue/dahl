#include "stdlib.h"
#include "utils.h"
#include <stdio.h>

#define LEARNING_RATE 0.1F
#define N_EPOCHS 200

void train_network(dataset* set, dahl_convolution* conv, dahl_pooling* pool, dahl_dense* dense, size_t batch_size, size_t n_classes)
{
    dahl_block* image_block = set->train_images;
    dahl_matrix* y_categorical = vector_to_categorical(set->train_labels, n_classes);

    dahl_shape2d gradients_shape = {
        .x = n_classes,
        .y = batch_size
    };

    dahl_matrix* gradients = matrix_init(gradients_shape);

    block_partition_along_z_batch(image_block, batch_size);
    matrix_partition_along_y_batch(y_categorical, batch_size);

    // size_t const n_samples = block_get_nb_children(image_block);
    size_t const n_samples = 5000; // block_get_shape(image_block).z
    size_t const n_batches_per_epoch = n_samples / batch_size; // Number of batch we want to do per epoch, not to be confused with batch size

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        printf("Epoch %lu\n", epoch);

        double total_loss = 0.0F;
        unsigned int correct_predictions = 0;

        for (size_t i = 0; i < n_batches_per_epoch; i++)
        {
            dahl_block const* image_batch = GET_SUB_BLOCK(image_block, i);
            dahl_matrix const* target_batch = GET_SUB_MATRIX(y_categorical, i);

            dahl_tensor* conv_out = convolution_forward(conv, image_batch);
            dahl_tensor* pool_out = pooling_forward(pool, conv_out);
            dahl_matrix* dense_out = dense_forward(dense, pool_out); // Returns the predictions for each batch

            total_loss += task_vector_cross_entropy_loss_batch(dense_out, target_batch);
            correct_predictions += task_check_predictions_batch(dense_out, target_batch);
            task_vector_cross_entropy_loss_gradient_batch(dense_out, target_batch, gradients);

            dahl_tensor* dense_back = dense_backward(dense, gradients, LEARNING_RATE);
            dahl_tensor* pool_back = pooling_backward(pool, dense_back);
            dahl_block* conv_back = convolution_backward(conv, pool_back, LEARNING_RATE);
            // Why aren't we using bacward convolution result?
        }

        printf("Average loss: %f - Accuracy: %f\%\n",
           total_loss / (dahl_fp)n_samples,
           correct_predictions / (dahl_fp)n_samples * 100);
    }

    block_unpartition(image_block);
    matrix_unpartition(y_categorical);
}

int main(int argc, char **argv)
{
    // Set the seed for reproducible results. Also it seems that StarPU might also set the seed somewhere,
    // because it still works when removing this line.
    srand(42);
    dahl_init();

    dataset* set = load_mnist(argv[1], argv[2]);

    size_t const num_channels = 1;
    size_t const num_classes = 10;
    size_t const filter_size = 6;
    size_t const pool_size = 2;
    // FIXME: support batch size that do not divide the dataset size
    size_t constexpr batch_size = 10;
    dahl_shape3d constexpr input_shape = { .x = 28, .y = 28, .z = batch_size };

    dahl_convolution* conv = convolution_init(input_shape, filter_size, num_channels);
    dahl_pooling* pool = pooling_init(pool_size, conv->output_shape);
    dahl_dense* dense = dense_init(pool->output_shape, num_classes);
    
    train_network(set, conv, pool, dense, batch_size, num_classes);

    dahl_shutdown();
    return 0;
}
