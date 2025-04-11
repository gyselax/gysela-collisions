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
