```
    (rocm-smi -r, ROCR_VISIBLE_DEVICES=3 HSA_XNACK=3, rest is default/toolchain specific):
    Koliop: v0.0.22
      KOLIOP_TOOLCHAIN_NAME: mi250x_usm.hipcc.adastra
      KOLIOP_BACKEND_IS_GPU_LIKE: true
      KOLIOP_BACKEND_IS_CPU_LIKE: true
      KOLIOP_ASSERT_ENABLED: true
      KOLIOP_FENCE_ON_OPERATOR_EXIT: false
      KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: true
      KOLIOP_BUILD_FORTRAN_INTERFACE: false (expecting f5d_s as (mu,vpar),(theta,r,phi),species)
      KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false
      KOLIOP_CONSTANT_MEMORY_USAGE: false
      koliop::collision::Constant::kDisjointExecutionSpaceCount: 1
      koliop::collision::Constant::kDiscardedStreamCount: 8
      koliop::collision::Constant::kDefaultExecutionSpacePoolIndex: 0
      koliop::collision::Constant::kPresetBlockExtent: [256,128,1] [128,    256,1] [128,128,2] [128,64,4] [64,128,4] [128,32,8] [128,128,1]     [128,64,2] [128,32,4] [64,128,2] [128,64,1] [128,32,2] [64,128,1]     [64,64,2] [32,128,2] [128,32,1] [64,64,1] [64,32,2] [32,128,1] [32,    64,2] [64,32,1] [32,64,1] [32,32,2] [32,32,1] [1,1,1]
      Kokkos Version: 4.6.0
    Compiler:
      KOKKOS_COMPILER_CLANG: 1800
    Architecture:
      CPU architecture: AMD_ZEN3
      Default Device: HIP
      GPU architecture: AMD_GFX90A
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
      Kernel reports HMM module via `CONFIG_HMM_MIRROR=y` in `/boot/    config`: yes
    macro  KOKKOS_ENABLE_HIP : defined
    macro  HIP_VERSION = 60342134 = version 6.3.42134
    macro KOKKOS_ENABLE_ROCTHRUST : defined
    macro KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC: yes
    Kokkos::HIP[ 0 ] gcnArch gfx90a:sramecc+:xnack+ : Selected
      Total Global Memory: 63.98 GiB
      Shared Memory per Block: 64 KiB
      APU or dGPU: dGPU
      Is Large Bar: 1
      Supports Managed Memory: 1
      Architecture capable of accessing system allocated memory: 1
      System allows accessing system allocated memory on GPU: 1
      Wavefront Size: 64
    0x21bb020
    BENCH:Warming up for 3 samples.
    BENCH:Warmed up in 37928 ms | 12642ms/sample
    BENCH:Benchmark epsilon (ns)            : 12642668
    BENCH:Running                           : ..........
    BENCH:Sample taken (iteration count)    : 10 (10)
    BENCH:Samples (ns)                      : 2116079208 2115904207     2114847552 2115765828 2116931476 2118147771 2117479241 2115534318     2116578149 2117117968
    BENCH:Samples (sorted) (ns)             : 2114847552 2115534318     2115765828 2115904207 2116079208 2116578149 2116931476 2117117968     2117479241 2118147771
    BENCH:Min|Max (ms)                      : 2114 | 2118
    BENCH:Average (ms)                      : 2116
    BENCH:Quantile (ms) at (0.25, 0.5, 0.75): (2115, 2116, 2117)
    Deinitialized()
```

NOTE: observe how the first iteration in the warmup are slow. We migrate the page to the GPU. During this phase, we observe the GPu's memory go up progressively, as pages are migrated. Once the pages are migrated the run time is significantly faster. That said, we are still faster by at most x0.765 compared to the plain hipMalloc version!
