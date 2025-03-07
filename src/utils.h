#ifndef DAHL_UTILS_H
#define DAHL_UTILS_H

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <starpu.h>

#include "types.h"

starpu_data_handle_t block_init(const shape3d shape);
void block_fill_random(starpu_data_handle_t handle);
void block_print_from_handle(starpu_data_handle_t handle);

starpu_data_handle_t matrix_init(const shape2d shape);
void matrix_fill_random(starpu_data_handle_t handle);
void matrix_print_from_handle(starpu_data_handle_t handle);

#endif //!DAHL_UTILS_H
