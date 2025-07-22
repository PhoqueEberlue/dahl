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

int read_int(FILE *file)
{
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
