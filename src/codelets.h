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

#endif //!DAHL_CODELETS_H
