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
    size_t const n_batches = n_samples / batch_size; // Number of batch we want to do per epoch, not to be confused with batch size

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        printf("Epoch %lu\n", epoch);

        double total_loss = 0.0F;
        float correct_predictions = 0;

        for (size_t i = 0; i < n_batches; i++)
        {
            dahl_block* images = block_get_sub_block(image_block, i); // Image batch
            dahl_matrix* targets = matrix_get_sub_matrix(y_categorical, i); // Target batch

            dahl_tensor* conv_out = convolution_forward(conv, images);
            dahl_tensor* pool_out = pooling_forward(pool, conv_out);
            dahl_matrix* dense_out = dense_forward(dense, pool_out); // predictions for each batch

            // TODO: probably nice to refactor cross entropy to accept a batch as an argument
            matrix_partition_along_y(dense_out);
            matrix_partition_along_y(targets);
            matrix_partition_along_y(gradients);

            for (size_t b = 0; b < matrix_get_nb_children(dense_out); b++)
            {
                dahl_vector const* sub_dense_out = matrix_get_sub_vector(dense_out, b);
                dahl_vector const* sub_targets = matrix_get_sub_vector(targets, b);
                dahl_vector* sub_gradients = matrix_get_sub_vector(gradients, b);

                dahl_fp loss = task_vector_cross_entropy_loss(sub_dense_out, sub_targets);
                total_loss += loss;

                if (check_prediction(sub_dense_out, sub_targets))
                {
                    correct_predictions += 1.0F;
                }
                
                // Compute the gradient
                task_vector_cross_entropy_loss_gradient(sub_dense_out, sub_targets, sub_gradients);
            }

            matrix_unpartition(dense_out);
            matrix_unpartition(targets);
            matrix_unpartition(gradients);

            dahl_tensor* dense_back = dense_backward(dense, gradients, LEARNING_RATE);
            dahl_tensor* pool_back = pooling_backward(pool, dense_back);
            dahl_block* conv_back = convolution_backward(conv, pool_back, LEARNING_RATE);
            // Why aren't we using bacward convolution result?
        }

        printf("Average loss: %f - Accuracy: %f\%\n",
           total_loss / (dahl_fp)n_samples,
           correct_predictions / (float)n_samples * 100.0F);
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
    size_t const pool_size = 16;
    size_t constexpr batch_size = 2;
    dahl_shape3d constexpr input_shape = { .x = 28, .y = 28, .z = batch_size };

    dahl_convolution* conv = convolution_init(input_shape, filter_size, num_channels);
    dahl_pooling* pool = pooling_init(pool_size, conv->output_shape);
    dahl_dense* dense = dense_init(pool->output_shape, num_classes);
    
    train_network(set, conv, pool, dense, batch_size, num_classes);

    dahl_shutdown();
    return 0;
}
