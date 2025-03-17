#include "utils.h"
#include <stdlib.h>

char* space_offset(size_t const offset)
{
    char* res = malloc((offset + 1) * sizeof(char));

    for (int i = 0; i<offset; i += 1)
    {
        res[i] = ' ';
    }

    res[offset] = '\0';

    return res;
}
