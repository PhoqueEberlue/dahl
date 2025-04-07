#ifndef DAHL_MNIST_LOADER_H
#define DAHL_MNIST_LOADER_H

#include "dahl_data.h"
#include <stdio.h>

// TODO: make it generic
typedef struct
{
    dahl_block* train_images;
    unsigned char *train_labels;
} dataset;

int read_int(FILE *file);
dahl_block* load_mnist_images(char const* filename);
unsigned char* load_mnist_labels(char const* filename);
dataset* load_mnist(char const* image_file, char const* label_file);
void free_dataset(dataset* set);

#endif //!DAHL_MNIST_LOADER_H
