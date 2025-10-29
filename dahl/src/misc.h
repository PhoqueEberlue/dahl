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

// Saturating substraction operation. If the result was supposed to underflow, return 0 instead.
#if defined (__i386__) || defined (__x86_64__)
inline size_t sub_sat(size_t a, size_t b)
{  
    size_t tmp = 0;

    // { AT&T    | Intel }  syntax alternatives.
    // The other versions without this  will break with -masm=intel
    __asm__("sub     { %[b],%[a]   | %[a],  %[b] }\n\t"
            "cmovnc  { %[a],%[dst] | %[dst],%[a] }"
            : [dst] "+r" (tmp), [a] "+&r" (a)
            : [b] "g" (b)
    );
    return tmp;
}
#else
// sub_sat equivalent without using inlined assembly. Not perfectly optimized though.
inline size_t sub_sat(size_t a, size_t b)
{
    size_t c = a - b;
    // This if can only happen if `c` underflow'd, thus the compiler can reuse overflow flag and
    // use a conditional move to execute c = 0;
    if (c > a)
        c = 0;
    return c;
}
#endif

#endif //!DAHL_MISC_H
