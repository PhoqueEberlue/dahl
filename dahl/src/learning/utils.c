#include "../../include/dahl_utils.h"
#include <stdio.h>
#include <stdlib.h>

void print_predictions_batch(dahl_matrix const* predictions_batch, dahl_matrix const* target_batch, 
        dahl_tensor const* image_batch, char const** labels)
{
    matrix_acquire(predictions_batch);
    matrix_acquire(target_batch);

    size_t const batch_size = matrix_get_shape(predictions_batch).y;
    size_t const num_classes = matrix_get_shape(predictions_batch).x;
    size_t const num_channels = tensor_get_shape(image_batch).z;

    dahl_tensor_p* image_batch_p = tensor_partition_along_t(image_batch, DAHL_READ);

    for (size_t y = 0; y < batch_size; y++)
    {
        dahl_block const* image = GET_SUB_BLOCK(image_batch_p, y);

        size_t max_index = 0;
        size_t true_index = 0;
        dahl_fp max_value = matrix_get_value(predictions_batch, max_index, y);

        for (size_t x = 0; x < num_classes; x++)
        {
           dahl_fp pred_val = matrix_get_value(predictions_batch, x, y);
           dahl_fp targ_val = matrix_get_value(target_batch, x, y);

           if (pred_val > max_value) 
           {
               max_index = x;
               max_value = pred_val;
           }
           if (targ_val == 1.0F) 
           {
               true_index = x;
           }
        }

        printf("Predicted class: %s (%lu)\n", labels[max_index], max_index);
        printf("True class: %s (%lu)\n", labels[true_index], true_index);

        // Display using ImageMagick
        if (num_channels == 1)
        {
            dahl_block_p* image_p = block_partition_along_z(image, DAHL_READ);
            dahl_matrix const* image_single_channel = GET_SUB_MATRIX(image_p, 0);
            matrix_image_display(image_single_channel, 10);

            // We can also display in ASCII in the terminal
            // matrix_print_ascii(image_single_channel, 0.5);
            block_unpartition(image_p);
        }
        else if (num_channels == 3)
        {
            block_image_display(image, 10);
        }
        else
        {
            printf("Error, wrong number of channels");
            exit(1);
        } 
    }

    tensor_unpartition(image_batch_p);

    matrix_release(predictions_batch);
    matrix_release(target_batch);
}
