// Small utility functions not to be exposed in the public API
#ifndef DAHL_MISC_H
#define DAHL_MISC_H

#include <stddef.h>
#include <stdio.h>

char* space_offset(const size_t offset);

// Function to read 4-byte integer from file (big-endian format)
int read_int(FILE *file);

FILE* temp_file_create(char* filename);
void temp_file_delete(char* filename, FILE* fp);

#endif //!DAHL_MISC_H
