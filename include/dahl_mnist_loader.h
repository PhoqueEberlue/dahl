#ifndef DAHL_MNIST_LOADER_H
#define DAHL_MNIST_LOADER_H

#include "dahl_data.h"
#include <stdio.h>

typedef struct
{
    dahl_block* train_images;
    dahl_vector* train_labels;
} dataset;

int read_int(FILE* file);
dahl_block* load_mnist_images(dahl_arena*, char const* filename);
dahl_vector* load_mnist_labels(dahl_arena*, char const* filename);
dataset* load_mnist(dahl_arena*, char const* image_file, char const* label_file);

#endif //!DAHL_MNIST_LOADER_H
