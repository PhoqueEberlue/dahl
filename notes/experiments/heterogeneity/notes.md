---
tldr: Comparing DAHL and Pytorch on GPU/CPU with different StarPU schedulers
commit: ``
---

┌─────────────────────────────────────────────────┬─────────────┐
│Parameters                                       │Metrics      │
├─────────┬─────────────┬───────────────┬─────────┤             │
│Framework│Dataset      │Processing Unit│Scheduler│             │
├─────────┼─────────────┼───────────────┼─────────┼─────────────┤
│DAHL     │Fashion MNIST│CPU only       │DM       │Energy       │
│Pytorch  │CIFAR-10     │GPU only       │DMDA     │Accuracy/Loss│
│         │Big Fashion  │CPU + GPU      │DMDAP    │%CPU usage   │
│         │             │               │DMDAR    │%GPU usage   │
│         │             │               │DMDAS    │Runtime      │
│         │             │               │DMDASD   │             │
└─────────┴─────────────┴───────────────┴─────────┴─────────────┘

-> 54 executions for DAHL, if we had non-performance modeling schedulers we could go to 99
executions.
-> 6 executions for pytorch
