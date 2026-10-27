```
    Naive benchmarks;
    On GPU: run rocm-smi -r before measuring.
    On CPU: clocked at 2.4 GHz.
                                                                        (ms)
                             64x128x64x128x32x2 | 1 GPU                                     |        64x128x32x32x2x2 | 1 thread
                             MI250X  MI250X (no erf)     MI300   MI300 (no erf) MI300 (USM)  GENOA (cce toolchain)   GENOA (no erf & cce toolchain) GENOA (aocc toolchain)
    MDRANGE                  1859    1712                815     750            x0.92-0.96   10508                   2464                           2963
    Flattened mod            1810    1727                781     740            x0.92-0.96   13086                   10021                          15751
    Flattened mod pow2       1734    1653                740     713            x0.92-0.96   12272                   9253                           15292
    Flattened div chain      1823    1744                782     747            x0.92-0.96   13071                   10028                          15773
    Flattened div chain pow2 1751    1670                750     721            x0.92-0.96   12341                   9481                           15307

    GENOA timings using different toolchains:
    AOCC 5.0       toolchain, 1 thread (no openmp):  2963ms // Everything vectorized (except reductions)
    CCE 18         toolchain, 1 thread (no openmp): 10508ms // Cpar          not vectorized
    AMDCLANG 6.2.1 toolchain, 1 thread (no openmp): 10787ms // Cpar          not vectorized
    OneAPI 2022    toolchain, 1 thread (no openmp): 16002ms // Cpar          not vectorized
    GCC 13.2.1     toolchain, 1 thread (no openmp): 13386ms // Cpar and more not vectorized
    ICC 2021       toolchain, 1 thread (no openmp): 22423ms // Crap ?
```