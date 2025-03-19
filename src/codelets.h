#ifndef DAHL_CODELETS_H
#define DAHL_CODELETS_H

#include <starpu.h>

// TODO: doc
void matrix_cross_correlation(void *buffers[3], void *cl_arg);

static struct starpu_perfmodel perf_model_matrix_cross_correlation =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_matrix_cross_correlation"
};
 
 
static struct starpu_codelet cl_matrix_cross_correlation =
{
    .cpu_funcs = { matrix_cross_correlation },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
    .model = &perf_model_matrix_cross_correlation
};

void matrix_max_pooling(void *buffers[2], void *cl_arg);

static struct starpu_perfmodel perf_model_matrix_max_pooling =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_matrix_max_pooling"
};
 
 
static struct starpu_codelet cl_matrix_max_pooling =
{
    .cpu_funcs = { matrix_max_pooling },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_W },
    .model = &perf_model_matrix_max_pooling
};

void relu(void *buffers[1], void *cl_arg);

static struct starpu_perfmodel perf_model_relu =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_relu"
};
 
 
static struct starpu_codelet cl_relu =
{
    .cpu_funcs = { relu },
    .nbuffers = 1,
    .modes = { STARPU_RW },
    .model = &perf_model_relu
};


// Sum the elements of a block over Z axis and fill output block (but will be considered as a matrix)
// arg: input, output
void block_sum_z_axis(void *buffers[2], void *cl_arg);

static struct starpu_perfmodel perf_model_block_sum_z_axis =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_block_sum_z_axis"
};
 
 
static struct starpu_codelet cl_block_sum_z_axis =
{
    .cpu_funcs = { block_sum_z_axis },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_W },
    .model = &perf_model_block_sum_z_axis
};

void scal(void *buffers[1], void *cl_arg);

static struct starpu_perfmodel perf_model_scal =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_scal"
};
 
 
static struct starpu_codelet cl_scal =
{
    .cpu_funcs = { scal },
    .nbuffers = 1,
    .modes = { STARPU_RW },
    .model = &perf_model_scal
};

void sub(void *buffers[2], void *cl_arg);

static struct starpu_perfmodel perf_model_sub =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_sub"
};
 
 
static struct starpu_codelet cl_sub =
{
    .cpu_funcs = { sub },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
    .model = &perf_model_sub
};

void add(void *buffers[2], void *cl_arg);

static struct starpu_perfmodel perf_model_add =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_add"
};
 
 
static struct starpu_codelet cl_add =
{
    .cpu_funcs = { add },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
    .model = &perf_model_add
};

#endif //!DAHL_CODELETS_H
