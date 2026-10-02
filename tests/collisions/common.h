#include <KOLIOP/collisions.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

void Initialize(bool   do_kokkos_initialization,
                       size_t a_host_thread_count,
                       size_t a_device_index,
                       bool   with_verbose_initialization) {
    if(do_kokkos_initialization == Kokkos::is_initialized()) {
        // Either we are asked to initialize Kokkos and it is already
        // initialized or we are asked not to initialize kokkos and it was
        // done already (XNOR).
        throw std::runtime_error("KOLIOP_STATUS_INITIALIZATION_FAILURE");
    }
    if(with_verbose_initialization) {
        std::cout << "Koliop: v" << KOLIOP_VERSION << "\n";
        std::cout << "  KOLIOP_TOOLCHAIN_NAME: " << KOLIOP_TOOLCHAIN_NAME << "\n";
#if defined(KOLIOP_BACKEND_IS_GPU_LIKE)
        std::cout << "  KOLIOP_BACKEND_IS_GPU_LIKE: true\n";
#else
        std::cout << "  KOLIOP_BACKEND_IS_GPU_LIKE: false\n";
#endif
#if defined(KOLIOP_BACKEND_IS_CPU_LIKE)
        std::cout << "  KOLIOP_BACKEND_IS_CPU_LIKE: true\n";
#else
        std::cout << "  KOLIOP_BACKEND_IS_CPU_LIKE: false\n";
#endif
#if defined(KOLIOP_ASSERT_ENABLED)
        std::cout << "  KOLIOP_ASSERT_ENABLED: true\n";
#else
        std::cout << "  KOLIOP_ASSERT_ENABLED: false\n";
#endif
        std::cout << "  Expecting f5d_s as (mu,vpar),(theta,r,phi),species\n";
        std::cout << "  Expecting input buffers on device\n";
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
        std::cout << "  KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: true\n";
#else
        std::cout << "  KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false\n";
#endif
#if defined(KOLIOP_CONSTANT_MEMORY_USAGE)
        std::cout << "  KOLIOP_CONSTANT_MEMORY_USAGE: true\n";
#else
        std::cout << "  KOLIOP_CONSTANT_MEMORY_USAGE: false\n";
#endif
        std::cout << "  koliop::collisions::Constant::kDisjointExecutionSpaceCount: " << koliop::collisions::Constant::kDisjointExecutionSpaceCount << "\n";
        std::cout << "  koliop::collisions::Constant::kDiscardedStreamCount: " << koliop::collisions::Constant::kDiscardedStreamCount << "\n";
        std::cout << "  koliop::collisions::Constant::kDefaultExecutionSpacePoolIndex: " << koliop::collisions::Constant::kDefaultExecutionSpacePoolIndex << "\n";
        std::cout << "  koliop::collisions::Constant::kPresetBlockExtent: ";
        for(std::size_t i = 0; i < koliop::collisions::Constant::kPresetBlockExtentCount; ++i) {
            std::cout << "[" << koliop::collisions::Constant::kPresetBlockExtent[i][0]
                      << "," << koliop::collisions::Constant::kPresetBlockExtent[i][1]
                      << "," << koliop::collisions::Constant::kPresetBlockExtent[i][2]
                      << "] ";
        }
        std::cout << "\n";
    }
    if(do_kokkos_initialization) {
        auto the_configuration = Kokkos::InitializationSettings()
                                     .set_device_id(static_cast<int>(a_device_index))
                                     .set_disable_warnings(!with_verbose_initialization);
        // If KOLIOP_SET_NUM_THREADS is set, use that value.
        auto* koliop_set_num_threads_string = std::getenv("KOLIOP_SET_NUM_THREADS");
        if(koliop_set_num_threads_string) {
            a_host_thread_count = std::stoi(koliop_set_num_threads_string);
        }
        if(a_host_thread_count > 0) {
            the_configuration.set_num_threads(a_host_thread_count);
        }
        Kokkos::initialize(the_configuration);
    }
    if(!Kokkos::is_initialized()) {
        throw std::runtime_error("KOLIOP_STATUS_INITIALIZATION_FAILURE");
    }
    if(with_verbose_initialization) {
        Kokkos::print_configuration(std::cout, true);
    }
    std::cout << std::flush;
}

void Deinitialize(bool do_kokkos_deinitialization) {
    if(do_kokkos_deinitialization) {
        if(!Kokkos::is_initialized()) {
            throw std::runtime_error("KOLIOP_STATUS_DEINITIALIZATION_FAILURE");
        }
        Kokkos::finalize();
        if(Kokkos::is_initialized()) {
            throw std::runtime_error("KOLIOP_STATUS_DEINITIALIZATION_FAILURE");
        }
    }
}

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
