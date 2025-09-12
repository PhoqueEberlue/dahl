#include "../../include/dahl_utils.h"
#include <stdio.h>

void print_predictions_batch(dahl_matrix const* predictions_batch, dahl_matrix const* target_batch, 
        dahl_tensor const* image_batch, char const** labels)
{
    matrix_print(predictions_batch);
    matrix_print(target_batch);

    dahl_fp const* preds = matrix_data_acquire(predictions_batch);
    dahl_fp const* targs = matrix_data_acquire(target_batch);

    size_t const batch_size = matrix_get_shape(predictions_batch).y;
    size_t const num_classes = matrix_get_shape(predictions_batch).x;

    tensor_partition_along_t(image_batch);

    for (size_t y = 0; y < batch_size; y++)
    {
        dahl_block const* image = GET_SUB_BLOCK(image_batch, y);

        size_t max_index = 0;
        // FIXME: really need to improve indexing of data structures, here I do (y*ld) but it's not obvious at all.
        dahl_fp max_value = preds[y*num_classes];
        size_t true_index = 0;

        for (size_t x = 0; x < num_classes; x++)
        {
           if (preds[(y*num_classes)+x] > max_value) 
           {
               max_index = x;
               max_value = preds[(y*num_classes)+x];
           }
           if (targs[(y*num_classes)+x] == 1.0F) 
           {
               true_index = x;
           }
        }

        printf("Predicted class: %s (%lu)\n", labels[max_index], max_index);
        printf("True class: %s (%lu)\n", labels[true_index], true_index);

        // Partition by image channel dimension, and only display one channel for now
        // TODO: sum the colors channels to display in the terminal in the future?
        block_partition_along_z(image);
        dahl_matrix const* image_single_channel = GET_SUB_MATRIX(image, 0);
        matrix_print_ascii(image_single_channel, 0.5);
        block_unpartition(image);
    }

    tensor_unpartition(image_batch);

    matrix_data_release(predictions_batch);
    matrix_data_release(target_batch);
}
