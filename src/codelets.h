#ifndef DAHL_CODELETS_H
#define DAHL_CODELETS_H

#include <starpu.h>

// buffer arguments: input, kernel, output 
// note that for now all arguments are of type block but should be passed as sublocks and will be considered as matrix in the function.
// I haven't find any way to "cast" sublocks into matrix sadly.
void cross_correlation_2d(void *buffers[3], void *cl_arg);

static struct starpu_perfmodel perf_model_cross_correlation_2d =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_cross_correlation_2d"
};
 
 
static struct starpu_codelet cl_cross_correlation_2d =
{
    .cpu_funcs = { cross_correlation_2d },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
    .model = &perf_model_cross_correlation_2d
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
void sum_z_axis(void *buffers[2], void *cl_arg);

static struct starpu_perfmodel perf_model_sum_z_axis =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "perf_model_sum_z_axis"
};
 
 
static struct starpu_codelet cl_sum_z_axis =
{
    .cpu_funcs = { sum_z_axis },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_W },
    .model = &perf_model_sum_z_axis
};

// Scale a block with cl_arg
// arg: input
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

// arg: block a, block b
// op: a -= b
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
