#include "../include/dahl.h"
#include <stdio.h>

#define LEARNING_RATE 0.01F
#define N_EPOCHS 200

void train_network(dataset* set, dahl_convolution* conv, dahl_pooling* pool)
{
    dahl_block* image_block = set->train_images;

    block_partition_along_z(image_block);

    for (size_t epoch = 0; epoch < N_EPOCHS; epoch++)
    {
        printf("Epoch %lu\n", epoch);

        double total_loss = 0.0F;
        int correct_predictions = 0;

        for (size_t i = 0; i < block_get_sub_matrix_nb(image_block); i++)
        {
            dahl_matrix* image = block_get_sub_matrix(image_block, i);

            dahl_block* conv_out = convolution_forward(conv, image);
            dahl_block* pool_out = pooling_forward(pool, conv_out);
        }
    }

    block_unpartition(image_block);
}

int main(int argc, char **argv)
{
    dahl_init();

    dataset* set = load_mnist(argv[1], argv[2]);

    dahl_shape2d input_shape = { .x = 28, .y = 28 };

    dahl_convolution* conv = convolution_init(input_shape, 6, 2);
    dahl_pooling* pool = pooling_init(2);
    // full = Fully_Connected(121, 10)
    
    train_network(set, conv, pool);

    dahl_shutdown();
    return 0;
}


