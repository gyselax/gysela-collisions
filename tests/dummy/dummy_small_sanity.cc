#include <KOLIOP/koliop.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.h"

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

    static constexpr size_t r_extent     = 16;
    static constexpr size_t theta_extent = 8;
    static constexpr size_t tor3_extent  = 4;

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
