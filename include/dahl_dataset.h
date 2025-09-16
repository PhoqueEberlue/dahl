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

#endif //!DAHL_MNIST_LOADER_H
