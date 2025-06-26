#include "stdlib.h"
#include "utils.h"

#define LEARNING_RATE 0.05F
#define N_EPOCHS 200

void train_network(dataset* set, dahl_convolution* conv, dahl_pooling* pool, dahl_dense* dense, size_t num_classes)
{
    dahl_block* image_block = set->train_images;
    dahl_matrix* y_categorical = vector_to_categorical(set->train_labels, num_classes);

    // size_t const n_samples = block_get_sub_matrix_nb(image_block);
    size_t const n_samples = 5000; // Let's only use the first 5k for now
    size_t const batch_size = 10;

    // Init in the persistent arena 
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_persistent_arena;

    dahl_matrix* gradients = matrix_init((dahl_shape2d) { .x = num_classes, .y = batch_size });
    dahl_vector* summed_gradients = vector_init(num_classes);

    // Then switch to previous context.
    dahl_context_arena = save_arena;

    block_partition_along_z(image_block);
    matrix_partition_along_y(y_categorical);

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        printf("Epoch %lu\n", epoch);

        double total_loss = 0.0F;
        float correct_predictions = 0;

        for (size_t i = 0; i < n_samples; i+= batch_size)
        {
            matrix_partition_along_y(gradients);

            for (size_t j = 0; j < batch_size; j++)
            {
                size_t input_index = i + j;
                dahl_matrix* image = block_get_sub_matrix(image_block, input_index);
                dahl_vector* targets = matrix_get_sub_vector(y_categorical, input_index);
                dahl_vector* gradient = matrix_get_sub_vector(gradients, j);

                dahl_block* conv_out = convolution_forward(conv, image);
                dahl_block* pool_out = pooling_forward(pool, conv_out);
                dahl_vector* dense_out = dense_forward(dense, pool_out); // predictions

                dahl_fp loss = task_vector_cross_entropy_loss(dense_out, targets);
                total_loss += loss;

                if (check_prediction(dense_out, targets))
                {
                    correct_predictions += 1.0F;
                }

                task_vector_cross_entropy_loss_gradient(dense_out, targets, gradient);
            }

            matrix_unpartition(gradients);

            task_matrix_sum_y_axis(gradients, summed_gradients);
            TASK_DIVIDE_SELF(summed_gradients, batch_size);

            dahl_block* dense_back = dense_backward(dense, summed_gradients, LEARNING_RATE);
            dahl_block* pool_back = pooling_backward(pool, dense_back);
            dahl_matrix* conv_back = convolution_backward(conv, pool_back, LEARNING_RATE);
            // Why aren't we using bacward convolution result result?
        }

        printf("Average loss: %f - Accuracy: %f\%\n", total_loss / (dahl_fp)n_samples, correct_predictions / (float)n_samples * 100.0F);
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

    dahl_shape2d constexpr img_shape = { .x = 28, .y = 28 };
    size_t const num_channels = 2;
    size_t const num_classes = 10;
    size_t const filter_size = 6;
    size_t const pool_size = 2;

    dahl_convolution* conv = convolution_init(img_shape, filter_size, num_channels);
    dahl_pooling* pool = pooling_init(pool_size, conv->output_shape);
    dahl_dense* dense = dense_init(pool->output_shape, num_classes);
    
    train_network(set, conv, pool, dense, num_classes);

    dahl_shutdown();
    return 0;
}
