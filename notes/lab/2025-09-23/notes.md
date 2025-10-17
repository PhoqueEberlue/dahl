---
tldr: Energy and training time comparison when changing the number of cpu count. DAHL is better than pytorch when a small workload is (too much) parallelized.
commit: `8f6c5f6032cc05884a53c5d81380ca73294309c0`
---

Commands used to run the benchmarks (here with 12 cpu workers):
```bash
# For DAHL
sudo STARPU_SCHED=dmda STARPU_NCPU=12 alumet-agent --plugins rapl,csv exec ./example-mnist ../datasets/fashion-mnist/train-images-idx3-ubyte ../datasets/fashion-mnist/train-labels-idx1-ubyte

# For the comparision with Pytorch. Preserve-env required using nix environment
sudo --preserve-env=PATH,PYTHONPATH alumet-agent --plugins rapl,csv exec -- torchrun --nproc_per_node=12 pytorch-version.py
```

Generated one of the first benchmarks of DAHL in comparison to Pytorch's performance with the following parameters:
20 Epoch, 6000 samples (quite low), Batch size 10, CPU count 1-4-8-12.

I used the gloo backend for pytorch.
Note that the loss seems to be bugged in this version.
Also this was my first time using alumet, energy values might be wrong.

What's interesting is that we obtain much better results with DAHL when we're attributing many CPUs.
This can be probably explained by the fact that we are using such a small workload compared to my computer performances.
Indeed, vanilla pytorch outperforms gloo pytorch by itself, so in anyways we wouldn't want to parallelize.
This does not give us a lot of interesting results on that, however its encouraging for the future because we can start
to see the advantages of using a TGCS, as it will smartly parallelize and used only the ressources needed
(Compared to pytorch gloo that will be forced to used as much CPU as requested, thus losing a lot of time parallelizing this small training task).
