#include "../include/dahl.h"
#include <stdio.h>

#define LEARNING_RATE 0.05F
#define N_EPOCHS 200

bool check_prediction(dahl_vector const* const predictions, dahl_vector const* const targets)
{
    dahl_fp* pred_data = ANY_DATA_ACQUIRE(predictions);
    dahl_fp* targ_data = ANY_DATA_ACQUIRE(targets);

    size_t num_classes = vector_get_len(predictions);

    dahl_fp max_val = 0.0F;
    size_t max_index = 0;

    for (size_t i = 0; i < num_classes; i++)
    {
        if (pred_data[i] > max_val)
        {
            max_val = pred_data[i];
            max_index = i;
        }
    }

    bool res = (bool)(targ_data[max_index] == 1);

    ANY_DATA_RELEASE(predictions);
    ANY_DATA_RELEASE(targets);

    return res;
}

void train_network(dataset* set, dahl_convolution* conv, dahl_pooling* pool, dahl_dense* dense)
{
    dahl_block* image_block = set->train_images;
    dahl_matrix* y_categorical = vector_as_categorical(set->train_labels, 10);

    block_partition_along_z(image_block);
    matrix_partition_along_y(y_categorical);

    // size_t const n_samples = block_get_sub_matrix_nb(image_block);
    size_t const n_samples = 5000; // Let's only use the first 5k for now

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        printf("Epoch %lu\n", epoch);

        double total_loss = 0.0F;
        float correct_predictions = 0;

        for (size_t i = 0; i < n_samples; i++)
        { 
            dahl_matrix* image = block_get_sub_matrix(image_block, i);
            dahl_vector* targets = matrix_get_sub_vector(y_categorical, i);

            dahl_block* conv_out = convolution_forward(conv, image);
            dahl_block* pool_out = pooling_forward(pool, conv_out);
            dahl_vector* dense_out = dense_forward(dense, pool_out); // predictions

            dahl_fp loss = task_vector_cross_entropy_loss(dense_out, targets);
            total_loss += loss;

            if (check_prediction(dense_out, targets))
            {
                correct_predictions += 1.0F;
            }

            dahl_vector* gradient = task_vector_cross_entropy_loss_gradient(dense_out, targets);

            dahl_block* dense_back = dense_backward(dense, gradient, LEARNING_RATE);
            dahl_block* pool_back = pooling_backward(pool, dense_back);
            // TODO: maybe save the image in the struct so we don't have to pass it?
            dahl_matrix* conv_back = convolution_backward(conv, pool_back, LEARNING_RATE, image);
            // Why aren't we using bacward convolution result result?
        }

        printf("Average loss: %f - Accuracy: %f\%\n", total_loss / (dahl_fp)n_samples, correct_predictions / (float)n_samples * 100.0F);
    }

    block_unpartition(image_block);
    matrix_unpartition(y_categorical);
}

int main(int argc, char **argv)
{
    dahl_init();

    dataset* set = load_mnist(argv[1], argv[2]);

    dahl_shape2d constexpr input_shape = { .x = 28, .y = 28 };

    size_t const num_channels = 2;
    dahl_convolution* conv = convolution_init(input_shape, 6, num_channels);
    dahl_pooling* pool = pooling_init(2, conv->output_shape);
    dahl_dense* dense = dense_init(pool->output_shape, 10);
    
    train_network(set, conv, pool, dense);

    dahl_shutdown();
    return 0;
}
