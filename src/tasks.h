#ifndef DAHL_TASKS_H
#define DAHL_TASKS_H

#include "types.h"
#include <starpu.h>

// Performs a x b = c, where:
// - x is the cross correlation operator
// - a, b and c are array pointers to contiguous matrix data
// - the shape of c must respect: c_nx = a_nx - b_nx + 1 and c_ny = a.y - b.y + 1
// - the shape of b should be smaller than the shape of a
//
// Matrices should be passed as sublocks // TODO: can improve that?
void task_cross_correlation_2d(const starpu_data_handle_t a, const starpu_data_handle_t b, const starpu_data_handle_t c);
void task_relu(starpu_data_handle_t in);
void task_sum_z_axis(const starpu_data_handle_t in, starpu_data_handle_t out);
void task_scal(starpu_data_handle_t in, dahl_fp factor);
void task_sub(starpu_data_handle_t a, const starpu_data_handle_t b);
void task_add(const starpu_data_handle_t a, const starpu_data_handle_t b);

#endif //!DAHL_TASKS_H
