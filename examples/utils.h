#include "../include/dahl.h"

// TODO: maybe put this somewhere in my API
bool check_prediction(dahl_vector const* predictions, dahl_vector const* targets)
{
    size_t num_classes = vector_get_len(predictions);

    dahl_fp* pred_data = vector_data_acquire(predictions);
    dahl_fp* targ_data = vector_data_acquire(targets);

    dahl_fp max_val = 0.0F;
    size_t max_index = 0;

    for (size_t i = 0; i < num_classes; i++)
    {
        if (pred_data[i] > max_val)
        {
            max_val = pred_data[i];
            max_index = i;
        }
    }

    bool res = (bool)(targ_data[max_index] == 1);

    vector_data_release(predictions);
    vector_data_release(targets);

    return res;
}
