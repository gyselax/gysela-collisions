#include <KOLIOP/koliop.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

template <typename Callable>
auto NaiveBenchmark(Callable&& a_callable,
                    int64_t    iteration_per_sample = 1,
                    int64_t    minimum_sample       = 2,
                    int64_t    maximum_sample       = 128) {
    const auto DoSample = [&](int64_t niteration) {
        auto time_point_0 = std::chrono::steady_clock::now();
        for(int64_t i = 0; i < niteration; ++i) {
            a_callable(i);
        }
        auto time_point_1 = std::chrono::steady_clock::now();
        return (time_point_1 - time_point_0).count();
    };

    if(maximum_sample < minimum_sample) {
        maximum_sample = minimum_sample;
    }

    const auto sample_duration_ns_to_iteration_ms = iteration_per_sample * 1'000'000;

    const auto warmup_iteration_factor = minimum_sample / 10 + 2;
    std::cout << "BENCH:Warming up for " << warmup_iteration_factor << " samples.\n"
              << std::flush;
    const auto warmup_duration_ns = DoSample(warmup_iteration_factor * iteration_per_sample);

    std::cout << "BENCH:Warmed up in " << (warmup_duration_ns / 1'000'000) << " ms | " << (warmup_duration_ns / (warmup_iteration_factor * sample_duration_ns_to_iteration_ms)) << "ms/sample\n"
              << std::flush;

    // Assuming that each op is made of an overhead o and real work w.
    // If we assume that an op ran n times has time:
    //  o + w * n
    // Then as n grows to infinity the op duration is:
    //  (o + w * n) / n, o tends to zero leaving w.

    const int64_t kEpsilon = (warmup_duration_ns * 1'000) / (warmup_iteration_factor * 1'000'000); // 1/1000 of warmup

    std::cout << "BENCH:Benchmark epsilon (ns)            : " << kEpsilon << "\n"
              << std::flush;

    std::vector<int64_t> sample_record;
    sample_record.resize(maximum_sample, 0);

    int64_t sample_sum              = 0;
    int64_t sample_average          = 0;
    int64_t previous_sample_average = 0;

    std::cout << "BENCH:Running                           : ";

    int64_t isample = 0;
    for(; isample < maximum_sample; ++isample) {
        std::cout << "." << std::flush;
        const auto duration_ns = DoSample(iteration_per_sample);

        sample_record[isample] = duration_ns;
        sample_sum             = duration_ns + sample_sum;
        sample_average         = sample_sum / (isample + 1);

        if(minimum_sample <= (isample + 1) &&
           (sample_average - previous_sample_average) < kEpsilon) {
            break;
        }

        std::swap(sample_average, previous_sample_average);
    }
    std::cout << "\n"
              << std::flush;
    {
        const auto sample_recorded = isample + 1;

        auto first_sample = std::begin(sample_record);
        auto last_sample  = first_sample + sample_recorded;

        std::cout << "BENCH:Sample taken (iteration count)    : "
                  << sample_recorded << " (" << (sample_recorded * iteration_per_sample) << ") \n";
        std::cout << "BENCH:Samples (ns)                      : ";
        for(auto first = first_sample; first != last_sample; ++first) {
            std::cout << (*first) << " ";
        }
        std::cout << "\n";

        std::sort(first_sample, last_sample);

        std::cout << "BENCH:Samples (sorted) (ns)             : ";
        for(auto first = first_sample; first != last_sample; ++first) {
            std::cout << (*first) << " ";
        }
        std::cout << "\n";

        std::cout << "BENCH:Min|Max (ms)                      : "
                  << ((*(first_sample + 0)) / sample_duration_ns_to_iteration_ms) << " | "
                  << ((*(first_sample + isample)) / sample_duration_ns_to_iteration_ms) << "\n";
        std::cout << "BENCH:Average (ms)                      : "
                  << (sample_average / sample_duration_ns_to_iteration_ms) << "\n";
        std::cout << "BENCH:Quantile (ms) at (0.25, 0.5, 0.75): ("
                  << ((*(first_sample + ((sample_recorded * 25) / 100))) / sample_duration_ns_to_iteration_ms) << ", "
                  << ((*(first_sample + ((sample_recorded * 50) / 100))) / sample_duration_ns_to_iteration_ms) << ", "
                  << ((*(first_sample + ((sample_recorded * 75) / 100))) / sample_duration_ns_to_iteration_ms) << ")\n";
    }
}

int Test1() {
    // NOTE: This function is used to test the GPU code. We pass the operator
    // dummy arrays that get copied to the GPU anyway (assuming
    // koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=OFF) thus being
    // disjoint (avoiding cache effects).

    ::koliop_Operator an_operator;

    static constexpr int8_t  collision_interspecies = true;
    static constexpr int8_t  SOL                    = false;
    static constexpr int8_t  LIM                    = false;
    static constexpr int64_t ir_SOL_separatrix      = 32;
    static constexpr double  dt                     = 50.0;

    static constexpr size_t mu_extent   = 64;
    static constexpr size_t vpar_extent = 128;

    static constexpr size_t r_extent     = 64;
    static constexpr size_t theta_extent = 128;

#if defined(KOLIOP_BUILD_FORTRAN_INTERFACE)
    static constexpr size_t tor1_extent = r_extent;
    static constexpr size_t tor2_extent = theta_extent;
#else
    static constexpr size_t tor1_extent = theta_extent;
    static constexpr size_t tor2_extent = r_extent;
#endif
    static constexpr size_t tor3_extent    = 1;
    static constexpr size_t species_extent = 1;

    static constexpr size_t global_domain_r_extent     = r_extent;
    static constexpr size_t global_domain_theta_extent = tor3_extent + 1;
    static constexpr size_t global_domain_r_offset     = 0;
    static constexpr size_t global_domain_theta_offset = 0;

    std::vector<double> a_big_dummy_array{};
    a_big_dummy_array.resize(mu_extent *
                                 vpar_extent *
                                 tor1_extent *
                                 tor2_extent *
                                 tor3_extent *
                                 species_extent,
                             1.0);

    std::vector<double> a_big_dummy_f5d_s{};
    a_big_dummy_f5d_s.resize(mu_extent *
                                 vpar_extent *
                                 tor1_extent *
                                 tor2_extent *
                                 tor3_extent *
                                 species_extent,
                             1.0);

    if(::koliop_Create(&an_operator,
                       mu_extent,
                       vpar_extent,
                       tor1_extent,
                       tor2_extent,
                       tor3_extent,
                       species_extent,
                       global_domain_r_extent,
                       global_domain_theta_extent,
                       global_domain_r_offset,
                       global_domain_theta_offset,
                       collision_interspecies,
                       SOL,
                       LIM,
                       ir_SOL_separatrix,
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       nullptr, // &mask_LIM[0][0],
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data()) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Create() failed\n";
        return -1;
    }

    std::cout << an_operator << "\n";

    ::koliop_ComputeMoments(an_operator,
                            a_big_dummy_f5d_s.data());

    ::koliop_PrecomputeNu(an_operator);

    ::koliop_ComputeMaxwellianEvolution(an_operator,
                                        dt,
                                        a_big_dummy_f5d_s.data());

    ::koliop_ComputeQparOnNT(an_operator,
                             a_big_dummy_f5d_s.data());

    ::koliop_ComputeCpar(an_operator,
                         dt,
                         a_big_dummy_f5d_s.data());

    static constexpr int32_t choice = 4;

    ::koliop_ComputeMaxwellianUnshiftedLoc(an_operator,
                                           a_big_dummy_f5d_s.data(),
                                           choice);

    ::koliop_Apply(an_operator,
                   dt,
                   a_big_dummy_f5d_s.data());

    ::koliop_Fence(an_operator);

    ::koliop_ComputeDensityCorrection(an_operator,
                                      a_big_dummy_f5d_s.data());

    ::koliop_ComputeUparTCorrection(an_operator,
                                    dt,
                                    a_big_dummy_f5d_s.data());

    ::koliop_ExtractRaw(an_operator,
                        KOLIOP_EXTRACTION_IDENTIFIER_n_s,
                        a_big_dummy_f5d_s.data());

    ::koliop_Fence(an_operator);

    if(::koliop_Release(an_operator) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Release() failed\n";
    }

    return 0;
}

int Test2() {
    // Naive benchmarks;
    // On GPU: run rocm-smi -r before measuring.
    // On CPU: clocked at 2.4 GHz.
    //                                                                     (ms)
    //                          64x128x64x128x32x2 | 1 GPU                                     |        64x128x32x32x2x2 | 1 thread
    //                          MI250X  MI250X (no erf)     MI300   MI300 (no erf) MI300 (USM)  GENOA (cce toolchain)   GENOA (no erf & cce toolchain) GENOA (aocc toolchain)
    // MDRANGE                  1859    1712                815     750            x0.92-0.96   10508                   2464                           2963
    // Flattened mod            1810    1727                781     740            x0.92-0.96   13086                   10021                          15751
    // Flattened mod pow2       1734    1653                740     713            x0.92-0.96   12272                   9253                           15292
    // Flattened div chain      1823    1744                782     747            x0.92-0.96   13071                   10028                          15773
    // Flattened div chain pow2 1751    1670                750     721            x0.92-0.96   12341                   9481                           15307

    // GENOA timings using different toolchains:
    // AOCC 5.0       toolchain, 1 thread (no openmp):  2963ms // Everything vectorized (except reductions)
    // CCE 18         toolchain, 1 thread (no openmp): 10508ms // Cpar          not vectorized
    // AMDCLANG 6.2.1 toolchain, 1 thread (no openmp): 10787ms // Cpar          not vectorized
    // OneAPI 2022    toolchain, 1 thread (no openmp): 16002ms // Cpar          not vectorized
    // GCC 13.2.1     toolchain, 1 thread (no openmp): 13386ms // Cpar and more not vectorized
    // ICC 2021       toolchain, 1 thread (no openmp): 22423ms // Crap ?

    // Latest MI250X results (use/merge branch origin/benchmark/no_copy at 417fd, ROCR_VISIBLE_DEVICES=3, rest is default/toolchain specific):
    // Koliop: v0.0.22
    //   KOLIOP_TOOLCHAIN_NAME: mi300a.hipcc.adastra
    //   KOLIOP_BACKEND_IS_GPU_LIKE: true
    //   KOLIOP_BACKEND_IS_CPU_LIKE: true
    //   KOLIOP_ASSERT_ENABLED: true
    //   KOLIOP_FENCE_ON_OPERATOR_EXIT: false
    //   KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: false
    //   KOLIOP_BUILD_FORTRAN_INTERFACE: false (expecting f5d_s as (mu,vpar),(theta,r,phi),species)
    //   KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false
    //   KOLIOP_CONSTANT_MEMORY_USAGE: true
    //   koliop::collision::Constant::kDisjointExecutionSpaceCount: 2
    //   koliop::collision::Constant::kDiscardedStreamCount: 8
    //   koliop::collision::Constant::kDefaultExecutionSpacePoolIndex: 1
    //   koliop::collision::Constant::kPresetBlockExtent: [128,128,1] [128,64,2] [128,32,4] [64,128,2] [128,64,1] [128,32,2] [64,128,1] [64,64,2] [32,128,2] [128,32,1] [64,64,1] [64,32,2] [32,128,1] [32,64,2] [64,32,1] [32,64,1] [32,32,2] [32,32,1] [1,1,1]
    // Kokkos::HIP::initialize WARNING: running kernels for MI300X (discrete GPU) on a MI300A (APU).
    //   Kokkos Version: 4.6.0
    // Compiler:
    //   KOKKOS_COMPILER_CLANG: 1800
    // Architecture:
    //   CPU architecture: AMD_ZEN4
    //   Default Device: HIP
    //   GPU architecture: none
    //   platform: 64bit
    // Atomics:
    // Vectorization:
    //   KOKKOS_ENABLE_PRAGMA_IVDEP: no
    //   KOKKOS_ENABLE_PRAGMA_LOOPCOUNT: no
    //   KOKKOS_ENABLE_PRAGMA_UNROLL: no
    //   KOKKOS_ENABLE_PRAGMA_VECTOR: no
    // Memory:
    // Options:
    //   KOKKOS_ENABLE_ASM: no
    //   KOKKOS_ENABLE_CXX17: yes
    //   KOKKOS_ENABLE_CXX20: no
    //   KOKKOS_ENABLE_CXX23: no
    //   KOKKOS_ENABLE_CXX26: no
    //   KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: no
    //   KOKKOS_ENABLE_HWLOC: no
    //   KOKKOS_ENABLE_LIBDL: yes
    // Host Serial Execution Space:
    //   KOKKOS_ENABLE_SERIAL: yes

    // Serial Runtime Configuration:
    // Device Execution Space:
    //   KOKKOS_ENABLE_HIP: yes
    // HIP Options:
    //   KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE: no

    // Runtime Configuration:
    //   XNACK environment variable set: no
    //   Kernel reports HMM module via `CONFIG_HMM_MIRROR=y` in `/boot/config`: yes
    // macro  KOKKOS_ENABLE_HIP : defined
    // macro  HIP_VERSION = 60342134 = version 6.3.42134
    // macro KOKKOS_ENABLE_ROCTHRUST : defined
    // macro KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC: yes
    // Kokkos::HIP[ 0 ] gcnArch gfx942:sramecc+:xnack- : Selected
    //   Total Global Memory: 128 GiB
    //   Shared Memory per Block: 64 KiB
    //   APU or dGPU: APU
    //   Is Large Bar: 1
    //   Supports Managed Memory: 1
    //   Architecture capable of accessing system allocated memory: 1
    //   System allows accessing system allocated memory on GPU: 0
    //   Wavefront Size: 64
    // 0x2501df0
    // BENCH:Warming up for 3 samples.
    // BENCH:Warmed up in 2189 ms | 729ms/sample
    // BENCH:Benchmark epsilon (ns)            : 729800
    // BENCH:Running                           : ..........
    // BENCH:Sample taken (iteration count)    : 10 (10)
    // BENCH:Samples (ns)                      : 722652235 726554689 718461148 728502261 724778648 720431041 721955561 721226745 729239326 722481374
    // BENCH:Samples (sorted) (ns)             : 718461148 720431041 721226745 721955561 722481374 722652235 724778648 726554689 728502261 729239326
    // BENCH:Min|Max (ms)                      : 718 | 729
    // BENCH:Average (ms)                      : 723
    // BENCH:Quantile (ms) at (0.25, 0.5, 0.75): (721, 722, 726)
    // Deinitialized()
    // x0.97 when HSA_XNACK=1 (same binary, just, enable USM support, even if not used)

    // Latest MI250X results (use/merge branch origin/benchmark/no_copy at 417fd, ROCR_VISIBLE_DEVICES=3, rest is default/toolchain specific):
    // Koliop: v0.0.22
    //   KOLIOP_TOOLCHAIN_NAME: mi250.hipcc.adastra
    //   KOLIOP_BACKEND_IS_GPU_LIKE: true
    //   KOLIOP_BACKEND_IS_CPU_LIKE: true
    //   KOLIOP_ASSERT_ENABLED: true
    //   KOLIOP_FENCE_ON_OPERATOR_EXIT: false
    //   KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: false
    //   KOLIOP_BUILD_FORTRAN_INTERFACE: false (expecting f5d_s as (mu,vpar),(theta,r,phi),species)
    //   KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false
    //   KOLIOP_CONSTANT_MEMORY_USAGE: false
    //   koliop::collision::Constant::kDisjointExecutionSpaceCount: 2
    //   koliop::collision::Constant::kDiscardedStreamCount: 8
    //   koliop::collision::Constant::kDefaultExecutionSpacePoolIndex: 1
    //   koliop::collision::Constant::kPresetBlockExtent: [128,128,1] [128,64,2] [128,32,4] [64,128,2] [128,64,1] [128,32,2] [64,128,1] [64,64,2] [32,128,2] [128,32,1] [64,64,1] [64,32,2] [32,128,1] [32,64,2] [64,32,1] [32,64,1] [32,32,2] [32,32,1] [1,1,1]
    // Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
    //   In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
    //   For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
    //   For unit testing set OMP_PROC_BIND=false

    //   Kokkos Version: 4.6.0
    // Compiler:
    //   KOKKOS_COMPILER_CLANG: 1800
    // Architecture:
    //   CPU architecture: AMD_ZEN3
    //   Default Device: HIP
    //   GPU architecture: AMD_GFX90A
    //   platform: 64bit
    // Atomics:
    // Vectorization:
    //   KOKKOS_ENABLE_PRAGMA_IVDEP: no
    //   KOKKOS_ENABLE_PRAGMA_LOOPCOUNT: no
    //   KOKKOS_ENABLE_PRAGMA_UNROLL: no
    //   KOKKOS_ENABLE_PRAGMA_VECTOR: no
    // Memory:
    // Options:
    //   KOKKOS_ENABLE_ASM: no
    //   KOKKOS_ENABLE_CXX17: yes
    //   KOKKOS_ENABLE_CXX20: no
    //   KOKKOS_ENABLE_CXX23: no
    //   KOKKOS_ENABLE_CXX26: no
    //   KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: no
    //   KOKKOS_ENABLE_HWLOC: no
    //   KOKKOS_ENABLE_LIBDL: yes
    // Host Parallel Execution Space:
    //   KOKKOS_ENABLE_OPENMP: yes

    // OpenMP Runtime Configuration:
    // Kokkos::OpenMP thread_pool_topology[ 1 x 8 x 1 ]
    // Device Execution Space:
    //   KOKKOS_ENABLE_HIP: yes
    // HIP Options:
    //   KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE: no

    // Runtime Configuration:
    //   XNACK environment variable set: no
    //   Kernel reports HMM module via `CONFIG_HMM_MIRROR=y` in `/boot/config`: yes
    // macro  KOKKOS_ENABLE_HIP : defined
    // macro  HIP_VERSION = 60342134 = version 6.3.42134
    // macro KOKKOS_ENABLE_ROCTHRUST : defined
    // macro KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC: yes
    // Kokkos::HIP[ 0 ] gcnArch gfx90a:sramecc+:xnack- : Selected
    //   Total Global Memory: 63.98 GiB
    //   Shared Memory per Block: 64 KiB
    //   APU or dGPU: dGPU
    //   Is Large Bar: 1
    //   Supports Managed Memory: 1
    //   Architecture capable of accessing system allocated memory: 1
    //   System allows accessing system allocated memory on GPU: 0
    //   Wavefront Size: 64
    // 0x2962b80
    // BENCH:Warming up for 3 samples.
    // BENCH:Warmed up in 5237 ms | 1745ms/sample
    // BENCH:Benchmark epsilon (ns)            : 1745931
    // BENCH:Running                           : ..........
    // BENCH:Sample taken (iteration count)    : 10 (10)
    // BENCH:Samples (ns)                      : 1757237085 1747725079 1753501953 1749396166 1731256631 1743808146 1748624354 1729044417 1746677083 1751066711
    // BENCH:Samples (sorted) (ns)             : 1729044417 1731256631 1743808146 1746677083 1747725079 1748624354 1749396166 1751066711 1753501953 1757237085
    // BENCH:Min|Max (ms)                      : 1729 | 1757
    // BENCH:Average (ms)                      : 1745
    // BENCH:Quantile (ms) at (0.25, 0.5, 0.75): (1743, 1748, 1751)
    // Deinitialized()
    // x0.966 when HSA_XNACK=1 (same binary, just, enable USM support, even if not used)

    // NOTE: This function is used to test the GPU code. We pass the operator
    // dummy arrays that get copied to the GPU anyway (assuming
    // koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=OFF) thus being
    // disjoint (avoiding cache effects).

    ::koliop_Operator an_operator;

    static constexpr int8_t collision_interspecies = true;
    static constexpr int8_t SOL                    = false;
    static constexpr int8_t LIM                    = false;
    static constexpr double dt                     = 50.0;

    static constexpr size_t mu_extent   = 64;
    static constexpr size_t vpar_extent = 128;

    static constexpr size_t r_extent     = 64;
    static constexpr size_t theta_extent = 128;
    static constexpr size_t tor3_extent  = 32;
    // static constexpr size_t r_extent     = 32;
    // static constexpr size_t theta_extent = 32;
    // static constexpr size_t tor3_extent  = 2;

#if defined(KOLIOP_BUILD_FORTRAN_INTERFACE)
    static constexpr size_t tor1_extent = r_extent;
    static constexpr size_t tor2_extent = theta_extent;
#else
    static constexpr size_t tor1_extent = theta_extent;
    static constexpr size_t tor2_extent = r_extent;
#endif
    static constexpr size_t species_extent = 2;

    static constexpr size_t global_domain_r_extent     = r_extent;
    static constexpr size_t global_domain_theta_extent = theta_extent + 1;
    static constexpr size_t global_domain_r_offset     = 0;
    static constexpr size_t global_domain_theta_offset = 0;

    static constexpr int64_t ir_SOL_separatrix = global_domain_r_extent;

    std::vector<double> a_big_dummy_array{};
    a_big_dummy_array.resize(mu_extent *
                                 vpar_extent *
                                 tor1_extent *
                                 tor2_extent *
                                 tor3_extent *
                                 species_extent,
                             1.0);

    std::vector<double> a_big_dummy_f5d_s{};
    a_big_dummy_f5d_s.resize(mu_extent *
                                 vpar_extent *
                                 tor1_extent *
                                 tor2_extent *
                                 tor3_extent *
                                 species_extent,
                             1.0);

    if(::koliop_Create(&an_operator,
                       mu_extent,
                       vpar_extent,
                       tor1_extent,
                       tor2_extent,
                       tor3_extent,
                       species_extent,
                       global_domain_r_extent,
                       global_domain_theta_extent,
                       global_domain_r_offset,
                       global_domain_theta_offset,
                       collision_interspecies,
                       SOL,
                       LIM,
                       ir_SOL_separatrix,
                       // NOTE: the cache locality effect implied by spoofing
                       // all these array as a single ptr is not visible on GPU.
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data(),
                       nullptr, // &mask_LIM[0][0],
                       a_big_dummy_array.data(),
                       a_big_dummy_array.data()) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Create() failed\n";
        return -1;
    }

    std::cout << an_operator << "\n";

    NaiveBenchmark([&](auto) {
        if(::koliop_Collision(an_operator,
                              dt,
                              a_big_dummy_f5d_s.data()) != KOLIOP_STATUS_SUCCESS) {
            std::cout << "::koliop_Collision() failed\n";
            std::exit(-1);
        }

        ::koliop_Fence(an_operator);
    });

    if(::koliop_Release(an_operator) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Release() failed\n";
    }

    return 0;
}

int main(int, char**) {
    if(::koliop_Initialize(KOLIOP_TRUE, 8, 0, KOLIOP_TRUE) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
        return -1;
    }

    // NOTE: Makes sure the kernel are on the device. Indeed the initial kernel
    // launch has a huge latency due to the kernel needing to be copied to the
    // GPU's memory.
    if(Test1() < 0) {
        return -1;
    }

    if(Test1() < 0) {
        return -1;
    }

    if(Test2() < 0) {
        return -1;
    }

    if(::koliop_Deinitialize(KOLIOP_TRUE) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Deinitialize() failed\n";
        return -1;
    }

    std::cout << "Deinitialized()\n";

    return 0;
}
