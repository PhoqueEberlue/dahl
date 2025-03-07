#ifndef DAHL_CODELETS_H
#define DAHL_CODELETS_H

#include <starpu.h>

// Note that for now it takes blocks and performs multiple cross_correlation on X axis
// buffer arguments: inputs, filters, output
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

#endif //!DAHL_CODELETS_H
