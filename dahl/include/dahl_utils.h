#ifndef DAHL_UTILS_H
#define DAHL_UTILS_H

#include "dahl_data.h"

void print_predictions_batch(dahl_matrix const* predictions_batch, dahl_matrix const* target_batch, dahl_tensor const* image_batch, char const** labels);

#endif //!DAHL_UTILS_H
