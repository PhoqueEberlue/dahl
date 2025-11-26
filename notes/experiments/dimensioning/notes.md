---
tldr: Comparing DAHL and Pytorch varying batch size and NCPU.
commit: ``
---
┌───────────────────────────────────────┬─────────────┐
│Parameters                             │Metrics      │
├─────────┬─────────────┬──────────┬────┤             │
│Framework│Dataset      │Batch size│NCPU│             │
├─────────┼─────────────┼──────────┼────┼─────────────┤
│DAHL     │Fashion MNIST│1         │1   │Energy       │
│Pytorch  │CIFAR-10     │16        │8   │Accuracy/Loss│
│         │Big Fashion  │64        │16  │%CPU usage   │
│         │             │256       │28  │Runtime      │
└─────────┴─────────────┴──────────┴────┴─────────────┘

-> 96 executions

