---
tldr: Implementing a "valid" size convolution with padding free buffers
commit: `edb0ae2381ccb2801872949a0db5afd6fa13a550`
---

Results with O0, not really interesting though:

Padded version 43-110 ms
Padding free version 50-170 ms

However, what really matters for the padding free version, is compilation with O3 optimizations,
because it completly removes conditionnal branches in the produced assembly for the sub_sat function
which is extremely useful in our for loops.
This works with saturating arithmetic that can be implemented very efficiently as such:

```c
size_t add_sat(size_t a, size_t b) {
    size_t c = a + b;
    if (c < a) /* Can only happen due to overflow */
        c = -1;
    return c;
}
```

which produces really nice assembly:
```asm
add_sat:
        mov     rax, rdi  # c, a
        mov     rdx, -1   # tmp92,
        add     rax, rsi  # c, b
        cmovc   rax, rdx  # c,, c, tmp92
        ret
```

This works because we can reuse some flags produce by `add` that indicates if c overflowed, so the if
can be simplified into using `cmovc`, see: https://www.felixcloutier.com/x86/cmovcc

However, sub function isn't as nicely compiled:

```c
size_t sub_sat(size_t a, size_t b)
{
    size_t c = a - b;
    if (c > a)
        c = 0;
    return c;
}
```

```asm
sub_sat:
        mov     rax, rdi  # c, a
        mov     edx, 0    # tmp92,
        sub     rax, rsi  # c, b
        cmp     rdi, rax  # a, c
        cmovb   rax, rdx  # c,, c, tmp92
        ret
```

Here it need an additional `cmp` instruction because the compiler is not able to guess to `c > a` is
equivalent to using the unsigned carry flag.
This may be sufficient, but if we really want, we could write inline assembly to obtain even better
results:

```c
size_t sub_sat_asm(size_t a, size_t b)
{
    size_t tmp = 0;
               // { AT&T    | Intel }  syntax alternatives.  The other versions without this  will break with -masm=intel
    asm ("sub     { %[b],%[a]   | %[a],  %[b] }\n\t"
         "cmovnc  { %[a],%[dst] | %[dst],%[a] }"
         : [dst] "+r" (tmp), [a] "+&r" (a)
         : [b] "g" (b)
         );
    return tmp;
}
```

```asm
sub_sat_asm:
        xor     eax, eax  # tmp
        sub     rdi, rsi  # a, b
        cmovnc  rax, rdi  # tmp, a
        ret
```


In O3 we obtain pretty close results in terms of pure codelet execution, however we gain clear
advantages: no need to perform copy and add padding, reduced number of task and synchronization,
better parallelization, less memory usage.

Results with O3:

Padded version ~20 ms
Padding free version ~10 ms

However it is clear that we could pre-compute ranges at model initialization and have clean loops.
