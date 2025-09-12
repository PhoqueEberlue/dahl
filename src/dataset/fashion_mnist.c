#include "../../include/dahl_dataset.h"
#include <assert.h>
#include <stdlib.h>
#include "../misc.h"
#include "unistd.h"

// Function to load MNIST images
dahl_tensor* load_mnist_images(dahl_arena* arena, char const* filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    size_t const nx = 28;
    size_t const ny = 28;
    size_t const channels = 1;
    size_t const samples = 60'000;
    size_t const n_classes = 10;

    // Read header
    int magic = read_int(file);
    size_t num_images = read_int(file);
    size_t rows = read_int(file);
    size_t cols = read_int(file);

    assert(num_images == samples);
    assert(rows == nx);
    assert(cols == ny);

    dahl_shape4d shape_image_tensor = { .x = nx, .y = ny, .z = channels, .t = samples };
    dahl_tensor* image_tensor = tensor_init(arena, shape_image_tensor);

    dahl_fp* images = tensor_data_acquire_mut(image_tensor);

    for (size_t t = 0; t < samples; t++)
    {
        // ignore channel dimension because there is only one
        for (size_t y = 0; y < ny; y++)
        {
            for (size_t x = 0; x < nx; x++)
            {
                unsigned char buffer;
                fread(&buffer, sizeof(unsigned char), 1, file);
                images[(t * nx * ny) + (y * nx) + x] = (dahl_fp)buffer / 255.0F;
            }
        }
    }

    tensor_data_release(image_tensor);
    fclose(file);

    printf("Loaded %lu images of size %lux%lu from %s\n", samples, nx, ny, filename);
    return image_tensor;
}

// Function to load MNIST labels
dahl_matrix* load_mnist_labels(dahl_arena* arena, char const* filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    // Read header
    int magic = read_int(file);
    size_t samples = read_int(file);

    size_t const n_classes = 10;

    dahl_shape2d shape_label = { 
        .x = n_classes,
        .y = samples,
    };

    dahl_matrix* label_vec = matrix_init(arena, shape_label);
    dahl_fp* labels = matrix_data_acquire_mut(label_vec);

    for (size_t i = 0; i < samples; i++)
    {
        unsigned char buffer;
        fread(&buffer, sizeof(unsigned char), 1, file);
        // Store the labels with a categorical format
        labels[(i * n_classes) + buffer] = 1;
    }

    fclose(file);
    matrix_data_release(label_vec);

    printf("Loaded %lu labels from %s\n", samples, filename);
    return label_vec;
}

dahl_dataset* dataset_load_fashion_mnist(dahl_arena* arena, char const* image_file, char const* label_file)
{
    dahl_dataset* res = dahl_arena_alloc(arena, sizeof(dahl_dataset));
    // Load training images & labels
    res->train_images = load_mnist_images(arena, image_file);
    res->train_labels = load_mnist_labels(arena, label_file); 

    return res;
}
