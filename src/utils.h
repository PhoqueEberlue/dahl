#ifndef DAHL_UTILS_H
#define DAHL_UTILS_H

#include <stddef.h>
#include <stdio.h>

char* space_offset(const size_t offset);

// Function to read 4-byte integer from file (big-endian format)
int read_int(FILE *file);

FILE* temp_file_create(char* filename);
FILE* temp_file_delete(char* filename, FILE* fp);

#endif //!DAHL_UTILS_H
