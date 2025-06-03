#include "../include/dahl_mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>

// Function to read 4-byte integer from file (big-endian format)
int read_int(FILE *file)
{
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Function to load MNIST images
dahl_block* load_mnist_images(char const* filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    // Read header
    int magic = read_int(file);
    size_t num_images = read_int(file);
    size_t rows = read_int(file);
    size_t cols = read_int(file);

    printf("Loaded %lu images of size %lux%lu from %s\n", num_images, rows, cols, filename);

    dahl_shape3d shape_image_block = { .x = rows, .y = cols, .z = num_images };
    dahl_block* image_block = block_init(shape_image_block);

    dahl_fp* images = block_data_acquire(image_block);

    for (size_t z = 0; z < num_images; z++)
    {
        for (size_t y = 0; y < cols; y++)
        {
            for (size_t x = 0; x < rows; x++)
            {
                unsigned char buffer;
                fread(&buffer, sizeof(unsigned char), 1, file);
                images[(z * cols * rows ) + (y* rows) + x] = (dahl_fp)buffer / 255.0F;
            }
        }
    }

    block_data_release(image_block);

    fclose(file);
    return image_block;
}

// Function to load MNIST labels
dahl_vector* load_mnist_labels(char const* filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    // Read header
    int magic = read_int(file);
    size_t num_labels = read_int(file);

    printf("Loaded %lu labels from %s\n", num_labels, filename);

    dahl_vector* label_vec = vector_init(num_labels);
    dahl_fp* labels = vector_data_acquire(label_vec);

    for (size_t i = 0; i < num_labels; i++)
    {
        unsigned char buffer;
        fread(&buffer, sizeof(unsigned char), 1, file);
        labels[i] = (dahl_fp)buffer;
    }

    fclose(file);
    vector_data_release(label_vec);

    return label_vec;
}

dataset* load_mnist(char const* image_file, char const* label_file)
{
    dataset* res = malloc(sizeof(dataset));
    // Load training images & labels
    res->train_images = load_mnist_images(image_file);
    res->train_labels = load_mnist_labels(label_file); 

    return res;
}

void free_dataset(dataset* set)
{
    block_finalize(set->train_images);
    vector_finalize(set->train_labels);
}
