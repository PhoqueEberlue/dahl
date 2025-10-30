#include "../codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../../include/dahl_types.h"
#include <assert.h>
#include <stddef.h>
#include <threads.h>

void scalar_accumulate(void *buffers[2], void *cl_arg)
{
	dahl_fp* dst = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[0]);
	dahl_fp const* src = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

	*dst += *src;
}
