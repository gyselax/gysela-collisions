```
    (use/merge branch origin/benchmark/no_copy at 417fd, rocm-smi -r, ROCR_VISIBLE_DEVICES=3, rest is default/toolchain specific):
    Koliop: v0.0.22
      KOLIOP_TOOLCHAIN_NAME: mi300a_usm.hipcc.adastra
      KOLIOP_BACKEND_IS_GPU_LIKE: true
      KOLIOP_BACKEND_IS_CPU_LIKE: true
      KOLIOP_ASSERT_ENABLED: true
      KOLIOP_FENCE_ON_OPERATOR_EXIT: false
      KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: true
      KOLIOP_BUILD_FORTRAN_INTERFACE: false (expecting f5d_s as (mu,vpar),(theta,r,phi),species)
      KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false
      KOLIOP_CONSTANT_MEMORY_USAGE: true
      koliop::collision::Constant::kDisjointExecutionSpaceCount: 2
      koliop::collision::Constant::kDiscardedStreamCount: 8
      koliop::collision::Constant::kDefaultExecutionSpacePoolIndex: 1
      koliop::collision::Constant::kPresetBlockExtent: [128,128,1] [128,64,2] [128,32,4] [64,128,2] [128,64,1] [128,32,2] [64,128,1] [64,64,2] [32,128,2] [128,32,1] [64,64,1] [64,32,2] [32,128,1] [32,64,2] [64,32,1] [32,64,1] [32,32,2] [32,32,1] [1,1,1]
      Kokkos Version: 4.6.0
    Compiler:
      KOKKOS_COMPILER_CLANG: 1800
    Architecture:
      CPU architecture: AMD_ZEN4
      Default Device: HIP
      GPU architecture: none
      platform: 64bit
    Atomics:
    Vectorization:
      KOKKOS_ENABLE_PRAGMA_IVDEP: no
      KOKKOS_ENABLE_PRAGMA_LOOPCOUNT: no
      KOKKOS_ENABLE_PRAGMA_UNROLL: no
      KOKKOS_ENABLE_PRAGMA_VECTOR: no
    Memory:
    Options:
      KOKKOS_ENABLE_ASM: no
      KOKKOS_ENABLE_CXX17: yes
      KOKKOS_ENABLE_CXX20: no
      KOKKOS_ENABLE_CXX23: no
      KOKKOS_ENABLE_CXX26: no
      KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: no
      KOKKOS_ENABLE_HWLOC: no
      KOKKOS_ENABLE_LIBDL: yes
    Host Serial Execution Space:
      KOKKOS_ENABLE_SERIAL: yes

    Serial Runtime Configuration:
    Device Execution Space:
      KOKKOS_ENABLE_HIP: yes
    HIP Options:
      KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE: no

    Runtime Configuration:
      XNACK environment variable set: yes
      Kernel reports HMM module via `CONFIG_HMM_MIRROR=y` in `/boot/config`: yes
    macro  KOKKOS_ENABLE_HIP : defined
    macro  HIP_VERSION = 60342134 = version 6.3.42134
    macro KOKKOS_ENABLE_ROCTHRUST : defined
    macro KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC: yes
    Kokkos::HIP[ 0 ] gcnArch gfx942:sramecc+:xnack+ : Selected
      Total Global Memory: 128 GiB
      Shared Memory per Block: 64 KiB
      APU or dGPU: APU
      Is Large Bar: 1
      Supports Managed Memory: 1
      Architecture capable of accessing system allocated memory: 1
      System allows accessing system allocated memory on GPU: 1
      Wavefront Size: 64
    0x17e1b10
    BENCH:Warming up for 3 samples.
    BENCH:Warmed up in 3118 ms | 1039ms/sample
    BENCH:Benchmark epsilon (ns)            : 1039600
    BENCH:Running                           : ..........
    BENCH:Sample taken (iteration count)    : 10 (10)
    BENCH:Samples (ns)                      : 747368667 747576608 746843264 748684935 750963688 749803202 750911738 748172111 745234124 745313665
    BENCH:Samples (sorted) (ns)             : 745234124 745313665 746843264 747368667 747576608 748172111 748684935 749803202 750911738 750963688
    BENCH:Min|Max (ms)                      : 745 | 750
    BENCH:Average (ms)                      : 748
    BENCH:Quantile (ms) at (0.25, 0.5, 0.75): (746, 748, 749)
    Deinitialized()
```

x0.963 compared to mi300a.hipcc.adastra

If the binding is wrong, you'll get significant slowdowns such as:

```
BENCH:Warming up for 3 samples.
BENCH:Warmed up in 29000 ms | 9666ms/sample
BENCH:Benchmark epsilon (ns)            : 9666772
BENCH:Running                           : ..........
BENCH:Sample taken (iteration count)    : 10 (10)
BENCH:Samples (ns)                      : 9677869928 9640414150 9621730421 9655388143 9696789920 9671605106 9628411745 9645312343 9648332640 9632526245
BENCH:Samples (sorted) (ns)             : 9621730421 9628411745 9632526245 9640414150 9645312343 9648332640 9655388143 9671605106 9677869928 9696789920
BENCH:Min|Max (ms)                      : 9621 | 9696
BENCH:Average (ms)                      : 9651
BENCH:Quantile (ms) at (0.25, 0.5, 0.75): (9632, 9648, 9671)
Deinitialized()
```