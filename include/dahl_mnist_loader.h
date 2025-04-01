#ifndef DAHL_MNIST_LOADER_H
#define DAHL_MNIST_LOADER_H

#include <stdio.h>
#include <stdlib.h>

#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28

// Function to read 4-byte integer from file (big-endian format)
int read_int(FILE *fp)
{
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fp);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Function to load MNIST images
unsigned char** load_mnist_images(char const* filename, size_t num_images)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    // Read header
    int magic = read_int(fp);
    num_images = read_int(fp);
    size_t rows = read_int(fp);
    size_t cols = read_int(fp);

    printf("Loaded %lu images of size %lux%lu from %s\n", num_images, rows, cols, filename);

    // Allocate memory
    unsigned char **images = (unsigned char**)malloc(num_images * sizeof(unsigned char*));
    for (size_t i = 0; i < num_images; i++) {
        images[i] = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
        fread(images[i], sizeof(unsigned char), rows * cols, fp);
    }

    fclose(fp);
    return images;
}

// Function to load MNIST labels
unsigned char* load_mnist_labels(char const* filename, size_t num_labels)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    // Read header
    int magic = read_int(fp);
    num_labels = read_int(fp);

    printf("Loaded %lu labels from %s\n", num_labels, filename);

    // Allocate memory
    unsigned char *labels = (unsigned char*)malloc(num_labels * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), num_labels, fp);

    fclose(fp);
    return labels;
}

typedef struct
{
    unsigned char **train_images;
    unsigned char *train_labels;
    size_t n_samples;
} dataset;

dataset* load_mnist(char const* image_file, char const* label_file)
{
    dataset* res = malloc(sizeof(dataset));
    // Load training images & labels
    res->n_samples = 60'000;
    res->train_images = load_mnist_images(image_file, res->n_samples);
    res->train_labels = load_mnist_labels(label_file, res->n_samples); 

    return res;
}

void free_dataset(dataset* set)
{
    for (int i = 0; i < set->n_samples; i++) {
        free(set->train_images[i]);
    }
    free(set->train_images);
    free(set->train_labels);
}

#endif //!DAHL_MNIST_LOADER_H
