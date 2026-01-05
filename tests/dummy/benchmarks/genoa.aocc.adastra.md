```
    (genoa.aocc.adastra/benchmark.sh):
    Koliop: v0.0.22
    KOLIOP_TOOLCHAIN_NAME: genoa.aocc.adastra
    KOLIOP_BACKEND_IS_GPU_LIKE: false
    KOLIOP_BACKEND_IS_CPU_LIKE: true
    KOLIOP_ASSERT_ENABLED: true
    KOLIOP_FENCE_ON_OPERATOR_EXIT: false
    KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: true
    KOLIOP_BUILD_FORTRAN_INTERFACE: false (expecting f5d_s as (mu,vpar),(theta,r,phi),species)
    KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false
    KOLIOP_CONSTANT_MEMORY_USAGE: false
    koliop::collision::Constant::kDisjointExecutionSpaceCount: 1
    koliop::collision::Constant::kDiscardedStreamCount: 0
    koliop::collision::Constant::kDefaultExecutionSpacePoolIndex: 0
    koliop::collision::Constant::kPresetBlockExtent: [256,128,1] [128,256,1] [128,128,2] [128,64,4] [64,128,4] [128,32,8] [128,128,1] [128,64,2] [128,32,4] [64,128,2] [128,64,1] [128,32,2] [64,128,1] [64,64,2] [32,128,2] [128,32,1] [64,64,1] [64,32,2] [32,128,1] [32,64,2] [64,32,1] [32,64,1] [32,32,2] [32,32,1] [1,1,1]
    Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
    In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
    For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
    For unit testing set OMP_PROC_BIND=false

    Kokkos Version: 4.6.0
    Compiler:
    KOKKOS_COMPILER_CLANG: 1706
    Architecture:
    CPU architecture: AMD_ZEN4
    Default Device: OpenMP
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
    Host Parallel Execution Space:
    KOKKOS_ENABLE_OPENMP: yes

    OpenMP Runtime Configuration:
    Kokkos::OpenMP thread_pool_topology[ 1 x 48 x 1 ]
    0x564f921faee0
    BENCH:Warming up for 3 samples.
    BENCH:Warmed up in 76045 ms | 25348ms/sample
    BENCH:Benchmark epsilon (ns)            : 25348423
    BENCH:Running                           : ..........
    BENCH:Sample taken (iteration count)    : 10 (10)
    BENCH:Samples (ns)                      : 25362198456 25031660956 24987068501 25205725968 25062700954 25005466821 25091847439 24804786214 25378373721 25240129306
    BENCH:Samples (sorted) (ns)             : 24804786214 24987068501 25005466821 25031660956 25062700954 25091847439 25205725968 25240129306 25362198456 25378373721
    BENCH:Min|Max (ms)                      : 24804 | 25378
    BENCH:Average (ms)                      : 25116
    BENCH:Quantile (ms) at (0.25, 0.5, 0.75): (25005, 25091, 25240)
    Deinitialized()
            Command being timed: "numactl -C 0-23,192-215 -- ./tests/demo_dummy_integration"
            User time (seconds): 14067.48
            System time (seconds): 690.53
            Percent of CPU this job got: 4409%
            Elapsed (wall clock) time (h:mm:ss or m:ss): 5:34.66
            Average shared text size (kbytes): 0
            Average unshared data size (kbytes): 0
            Average stack size (kbytes): 0
            Average total size (kbytes): 0
            Maximum resident set size (kbytes): 69156220
            Average resident set size (kbytes): 0
            Major (requiring I/O) page faults: 148
            Minor (reclaiming a frame) page faults: 6189742
            Voluntary context switches: 8207
            Involuntary context switches: 1391524
            Swaps: 0
            File system inputs: 0
            File system outputs: 0
            Socket messages sent: 0
            Socket messages received: 0
            Signals delivered: 0
            Page size (bytes): 4096
            Exit status: 0
    Energy delta: 122769
```
