---
tldr: Compared using a single matrix matrix multiplication vs using multiple matrix vector products
over a batch.
commit: `d496ae94d294a95e9d50dd52e37fbb1770db279c`
---

Indeed, it is worth it to split the matrix matrix mult into multiple matrix vector products because
they are able to get parallelized on each core. This is especially useful when using a large batch
size, because we would have however a single, linearly-growing task with the mat mat mult.
