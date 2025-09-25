#include "../../include/dahl_dataset.h"
#include <stdio.h>
#include <stdlib.h>

char const* FACTICE_CLASS_NAMES[10] = { "mais", "ou", "est", "donc", "or", "ni", "car", "bus", "voiture", "kamoulox" };

dahl_dataset* dataset_load_factice(dahl_arena* arena, dahl_shape3d images_shape, size_t num_samples)
{
    dahl_dataset* res = dahl_arena_alloc(arena, sizeof(dahl_dataset));

    size_t const n_classes = 10;

    dahl_shape4d image_set_shape = { 
        .x = images_shape.x,
        .y = images_shape.y,
        .z = images_shape.z,
        .t = num_samples
    };

    dahl_tensor* image_set = tensor_init(arena, image_set_shape); //, 0, 1);

    dahl_shape2d shape_label = { 
        .x = n_classes,
        .y = num_samples,
    };

    dahl_matrix* label_set = matrix_init(arena, shape_label);

    matrix_acquire(label_set);

    for (size_t y = 0; y < num_samples; y++)
    {
        // Randomly set one class to be the true class
        matrix_set_value(label_set, rand() % 10, y, 1);
    }

    matrix_release(label_set);

    res->train_images = image_set;
    res->train_labels = label_set;
    res->class_names = FACTICE_CLASS_NAMES;
    res->num_classes = n_classes;

    printf("Loaded %lu factice images and labels\n", num_samples);
    return res;
}
