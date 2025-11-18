#include "../../include/dahl_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../misc.h"

// Using master categories
size_t static constexpr n_classes = 7;
char const* BIG_FASHION_CLASS_NAMES[n_classes] = { 
    "Apparel", "Accessories", "Footwear", "Personal Care", "Free Items", "Sporting Goods", "Home" };

size_t category_to_index(char const* category)
{
    for (size_t i = 0; i < n_classes; i++)
    {
        if (strcmp(category, BIG_FASHION_CLASS_NAMES[i]) == 0)
            return i;
    }
    fprintf(stderr, "category %s, not found\n", category);
    exit(1);
}

dahl_dataset* dataset_load_big_fashion(
        dahl_arena* arena, char const* style_csv, char const* image_directory)
{
    FILE* style = fopen(style_csv, "r");    
    if (!style)
    {
        fprintf(stderr, "Could not open %s\n", style);
        return NULL;
    }

    dahl_dataset* res = dahl_arena_alloc(arena, sizeof(dahl_dataset));

    // Image properties
    size_t const nx = 270;
    size_t const ny = 360;
    size_t const channels = 3;
    // We have 44 441 samples but this represents:
    // 44441*1440*1080*3*sizeof(double)*10^-9 ~= 1658 Gigabyte.
    // So let's use less :)
    size_t const samples = 600; 

    dahl_shape4d shape_images = { 
        .x = nx,
        .y = ny,
        .z = channels,
        .t = samples
    };

    dahl_tensor* image_set = tensor_init(arena, shape_images);
    tensor_partition_along_t(image_set, DAHL_MUT);

    dahl_shape2d shape_label = { 
        .x = n_classes,
        .y = samples,
    };

    dahl_matrix* label_set = matrix_init(arena, shape_label);
    matrix_acquire_mut(label_set);

    char buffer[250];
    size_t n_sample = 0;

    // ignore header
    fgets(buffer, 250, style);
    size_t progress = 0;

    // For each sample
    for (size_t index_sample = 0; index_sample < samples; index_sample++)
    {
        fgets(buffer, 250, style);

        // Value of one cell
        char *id = strtok(buffer, ",");
        if (!id) { fprintf(stderr, "Couldn't read id"); }

        // Ignore Gender column
        strtok(NULL, ",");

        // Retrieve master category
        char* category = strtok(NULL, ",");
        if (!category) { fprintf(stderr, "Couldn't read category"); }

        dahl_block* image = GET_SUB_BLOCK_MUT(image_set, index_sample);

        char image_path[250];
        int ret = snprintf(image_path, sizeof(image_path), "%s%s.jpg", image_directory, id);
        assert(ret > 0);
        
        // Write jpg image into image buffer
        block_read_jpeg(image, image_path);

        // Write category in one-hot label format
        matrix_set_value(label_set, category_to_index(category), index_sample, 1);

        size_t new_progress = index_sample * 100 / samples;
        if (progress != new_progress)
        {
            printf("Loading dataset: %lu\%\n", new_progress);
            progress = new_progress;
        }
    }

    tensor_unpartition(image_set);
    matrix_release(label_set);
    
    res->train_images = image_set;
    res->train_labels = label_set;
    res->class_names = BIG_FASHION_CLASS_NAMES;
    res->num_classes = n_classes;

    // printf("Loaded %lu images and labels from %s\n", samples, data_batch_file);
    return res;
}
