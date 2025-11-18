---
tldr: First comparisons after implementing the GPU version.
commit: `cdf0d11ad7e23d81fc847756eba9b810e3312312`
---

Running on CPU+GPU
```bash
andrew@rammus ~/d/d/build (main) [nix] > time STARPU_SCHED=dmda ./basic_cnn ../datasets/fashion-mnist/train-images-idx3-ubyte ../datasets/fashion-mnist/train-labels-idx1-ubyte 
Epoch: 0, Loss: 2.282767, Accuracy: 0.122333
Epoch: 1, Loss: 2.036433, Accuracy: 0.312333
Epoch: 2, Loss: 1.414622, Accuracy: 0.604167
Epoch: 3, Loss: 1.026908, Accuracy: 0.665333
Epoch: 4, Loss: 0.884220, Accuracy: 0.694500
Epoch: 5, Loss: 0.812203, Accuracy: 0.714333
Epoch: 6, Loss: 0.765977, Accuracy: 0.728500
Epoch: 7, Loss: 0.732359, Accuracy: 0.741000
Epoch: 8, Loss: 0.706054, Accuracy: 0.750833
Epoch: 9, Loss: 0.684477, Accuracy: 0.759833
Epoch: 10, Loss: 0.666200, Accuracy: 0.767167
Epoch: 11, Loss: 0.650360, Accuracy: 0.770833
Epoch: 12, Loss: 0.636400, Accuracy: 0.775667
Epoch: 13, Loss: 0.623933, Accuracy: 0.781000
Epoch: 14, Loss: 0.612677, Accuracy: 0.785000
Epoch: 15, Loss: 0.602429, Accuracy: 0.788500
Epoch: 16, Loss: 0.593033, Accuracy: 0.792833
Epoch: 17, Loss: 0.584374, Accuracy: 0.797833
Epoch: 18, Loss: 0.576360, Accuracy: 0.798500
Epoch: 19, Loss: 0.568928, Accuracy: 0.801500
________________________________________________________
Executed in   47.16 secs    fish           external
   usr time  107.97 secs    2.85 millis  107.96 secs
   sys time   13.72 secs    0.97 millis   13.72 secs
```

Running on GPU only
```bash
andrew@rammus ~/d/d/build (main) [nix] > time STARPU_SCHED=dmda STARPU_NCPU=0 ./basic_cnn ../datasets/fashion-mnist/train-images-idx3-ubyte ../datasets/fashion-mnist/train-labels-idx1-ubyte
Epoch: 0, Loss: 2.284991, Accuracy: 0.118833
Epoch: 1, Loss: 2.035054, Accuracy: 0.315500
Epoch: 2, Loss: 1.408688, Accuracy: 0.607667
Epoch: 3, Loss: 1.019954, Accuracy: 0.669667
Epoch: 4, Loss: 0.876683, Accuracy: 0.702500
Epoch: 5, Loss: 0.805008, Accuracy: 0.719333
Epoch: 6, Loss: 0.759466, Accuracy: 0.732833
Epoch: 7, Loss: 0.726574, Accuracy: 0.746500
Epoch: 8, Loss: 0.700983, Accuracy: 0.754167
Epoch: 9, Loss: 0.680127, Accuracy: 0.763167
Epoch: 10, Loss: 0.662610, Accuracy: 0.768167
Epoch: 11, Loss: 0.647586, Accuracy: 0.772333
Epoch: 12, Loss: 0.634504, Accuracy: 0.775833
Epoch: 13, Loss: 0.622984, Accuracy: 0.780000
Epoch: 14, Loss: 0.612742, Accuracy: 0.783167
Epoch: 15, Loss: 0.603559, Accuracy: 0.787833
Epoch: 16, Loss: 0.595266, Accuracy: 0.791333
Epoch: 17, Loss: 0.587722, Accuracy: 0.795000
Epoch: 18, Loss: 0.580816, Accuracy: 0.798167
Epoch: 19, Loss: 0.574455, Accuracy: 0.801667

________________________________________________________
Executed in   83.13 secs    fish           external
   usr time  103.45 secs    3.67 millis  103.45 secs
   sys time    2.67 secs    0.00 millis    2.67 secs
```

Running on CPU only
```bash
andrew@rammus ~/d/d/build (main) [nix] > time STARPU_SCHED=dmda STARPU_NCUDA=0 ./basic_cnn ../datasets/fashion-mnist/train-images-idx3-ubyte ../datasets/fashion-mnist/train-labels-idx1-ubyte
Epoch: 0, Loss: 2.282767, Accuracy: 0.122333
Epoch: 1, Loss: 2.036433, Accuracy: 0.312333
Epoch: 2, Loss: 1.414622, Accuracy: 0.604167
Epoch: 3, Loss: 1.026908, Accuracy: 0.665333
Epoch: 4, Loss: 0.884220, Accuracy: 0.694500
Epoch: 5, Loss: 0.812203, Accuracy: 0.714333
Epoch: 6, Loss: 0.765977, Accuracy: 0.728500
Epoch: 7, Loss: 0.732359, Accuracy: 0.741000
Epoch: 8, Loss: 0.706054, Accuracy: 0.750833
Epoch: 9, Loss: 0.684477, Accuracy: 0.759833
Epoch: 10, Loss: 0.666200, Accuracy: 0.767167
Epoch: 11, Loss: 0.650360, Accuracy: 0.770833
Epoch: 12, Loss: 0.636400, Accuracy: 0.775667
Epoch: 13, Loss: 0.623933, Accuracy: 0.781000
Epoch: 14, Loss: 0.612677, Accuracy: 0.785000
Epoch: 15, Loss: 0.602429, Accuracy: 0.788500
Epoch: 16, Loss: 0.593033, Accuracy: 0.792833
Epoch: 17, Loss: 0.584374, Accuracy: 0.797833
Epoch: 18, Loss: 0.576360, Accuracy: 0.798500
Epoch: 19, Loss: 0.568928, Accuracy: 0.801500

________________________________________________________
Executed in   45.82 secs    fish           external
   usr time  103.55 secs    1.96 millis  103.55 secs
   sys time   13.31 secs    1.72 millis   13.31 secs

andrew@rammus ~/d/d/build (main) [nix] > 
```

How weird, absolute no gain running on CPU+GPU compared to CPU only, AND on top of that, GPU version
is wayyyy slower, by almost two times? Wow.
