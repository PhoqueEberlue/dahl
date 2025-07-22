#include "../../include/dahl_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include "../utils.h"

dahl_dataset* dataset_load_cifar_10(dahl_arena* arena, char const* data_batch_file)
{
    FILE *file = fopen(data_batch_file, "rb");
    if (!file)
    {
        printf("Error opening file: %s\n", data_batch_file);
        return NULL;
    }

    dahl_dataset* res = dahl_arena_alloc(arena, sizeof(dahl_dataset));

    // See: https://www.cs.toronto.edu/~kriz/cifar.html
    size_t const nx = 32;
    size_t const ny = 32;
    size_t const channels = 3;
    size_t const samples = 10'000;
    size_t const n_classes = 10;

    dahl_shape4d shape_images = { 
        .x = nx,
        .y = ny,
        .z = channels,
        .t = samples
    };

    dahl_tensor* image_set = tensor_init(arena, shape_images);
    dahl_fp* images = tensor_data_acquire_mut(image_set);

    dahl_shape2d shape_label = { 
        .x = n_classes,
        .y = samples,
    };

    dahl_matrix* label_set = matrix_init(arena, shape_label);
    dahl_fp* labels = matrix_data_acquire_mut(label_set);

    unsigned char buffer;

    for (size_t t = 0; t < samples; t++)
    {
        // Read the label
        fread(&buffer, sizeof(unsigned char), 1, file);
        // Store labels categorical format
        labels[(t * n_classes) + buffer] = 1;

        for (size_t z = 0; z < channels; z++)
        {
            for (size_t y = 0; y < ny; y++)
            {
                for (size_t x = 0; x < nx; x++)
                {
                    fread(&buffer, sizeof(unsigned char), 1, file);
                    images[(t * nx * ny * channels) + (z * nx * ny ) + (y * nx) + x] = (dahl_fp)buffer / 255.0F;
                }
            }
        }
    }

    tensor_data_release(image_set);
    matrix_data_release(label_set);

    res->train_images = image_set;
    res->train_labels = label_set;

    printf("Loaded %lu images and labels from %s\n", samples, data_batch_file);
    return res;
}
