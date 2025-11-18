#ifndef DAHL_MNIST_LOADER_H
#define DAHL_MNIST_LOADER_H

#include "dahl_data.h"
#include <stdio.h>

typedef struct
{
    dahl_tensor* train_images;
    dahl_matrix* train_labels;
    char const** class_names;
    size_t num_classes;
} dahl_dataset;

dahl_dataset* dataset_load_fashion_mnist(dahl_arena*, char const* image_file, char const* label_file);
dahl_dataset* dataset_load_cifar_10(dahl_arena* arena, char const* data_batch_file);
dahl_dataset* dataset_load_big_fashion(dahl_arena* arena, char const* style_csv, char const* image_directory);

// Load a factice dataset
dahl_dataset* dataset_load_factice(dahl_arena* arena, dahl_shape3d images_shape, size_t num_samples);

#endif //!DAHL_MNIST_LOADER_H
