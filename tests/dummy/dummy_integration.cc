#include <KOLIOP/koliop.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.h"

int Test1() {
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

    const size_t buffer_size = mu_extent * vpar_extent * tor1_extent * tor2_extent * tor3_extent * species_extent;

    Kokkos::View<double*> a_buffer_0_device_view("device_buffer", buffer_size);
    const auto            a_buffer_0_host_mirror = Kokkos::create_mirror_view(a_buffer_0_device_view);

    Kokkos::View<double*> a_buffer_1_device_view("device_buffer", buffer_size);
    const auto            a_buffer_1_host_mirror = Kokkos::create_mirror_view(a_buffer_0_device_view);

    for(size_t i = 0; i < buffer_size; ++i) {
        a_buffer_0_host_mirror(i) = 1.0;
        a_buffer_1_host_mirror(i) = 1.0;
    }

    Kokkos::deep_copy(a_buffer_0_device_view, a_buffer_0_host_mirror);
    Kokkos::deep_copy(a_buffer_1_device_view, a_buffer_1_host_mirror);

#if defined(KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
    std::cout << "Using device buffers, KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=ON" << std::endl;
    auto* a_buffer_0 = a_buffer_0_device_view.data();
    auto* a_buffer_1 = a_buffer_1_device_view.data();
#else
    std::cout << "Using host buffers, KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=OFF" << std::endl;
    auto* a_buffer_0 = a_buffer_0_host_mirror.data();
    auto* a_buffer_1 = a_buffer_1_host_mirror.data();
#endif

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
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       nullptr, // &mask_LIM[0][0],
                       a_buffer_0,
                       a_buffer_0) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Create() failed\n";
        return -1;
    }

    std::cout << an_operator << "\n";

    ::koliop_ComputeMoments(an_operator,
                            a_buffer_1);

    ::koliop_PrecomputeNu(an_operator);

    ::koliop_ComputeMaxwellianEvolution(an_operator,
                                        dt,
                                        a_buffer_1);

    ::koliop_ComputeQparOnNT(an_operator,
                             a_buffer_1);

    ::koliop_ComputeCpar(an_operator,
                         dt,
                         a_buffer_1);

    static constexpr int32_t choice = 4;

    ::koliop_ComputeMaxwellianUnshiftedLoc(an_operator,
                                           a_buffer_1,
                                           choice);

    ::koliop_Apply(an_operator,
                   dt,
                   a_buffer_1);

    ::koliop_Fence(an_operator);

    ::koliop_ComputeDensityCorrection(an_operator,
                                      a_buffer_1);

    ::koliop_ComputeUparTCorrection(an_operator,
                                    dt,
                                    a_buffer_1);

    ::koliop_ExtractRaw(an_operator,
                        KOLIOP_EXTRACTION_IDENTIFIER_n_s,
                        a_buffer_1);

    ::koliop_Fence(an_operator);

    if(::koliop_Release(an_operator) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Release() failed\n";
    }

    return 0;
}

int Test2() {
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

    const size_t buffer_size = mu_extent * vpar_extent * tor1_extent * tor2_extent * tor3_extent * species_extent;

    Kokkos::View<double*> a_buffer_0_device_view("device_buffer", buffer_size);
    const auto            a_buffer_0_host_mirror = Kokkos::create_mirror_view(a_buffer_0_device_view);

    Kokkos::View<double*> a_buffer_1_device_view("device_buffer", buffer_size);
    const auto            a_buffer_1_host_mirror = Kokkos::create_mirror_view(a_buffer_0_device_view);

    for(size_t i = 0; i < buffer_size; ++i) {
        a_buffer_0_host_mirror(i) = 1.0;
        a_buffer_1_host_mirror(i) = 1.0;
    }

    Kokkos::deep_copy(a_buffer_0_device_view, a_buffer_0_host_mirror);
    Kokkos::deep_copy(a_buffer_1_device_view, a_buffer_1_host_mirror);

#if defined(KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
    std::cout << "Using device buffers, KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=ON" << std::endl;
    auto* a_buffer_0 = a_buffer_0_device_view.data();
    auto* a_buffer_1 = a_buffer_1_device_view.data();
#else
    std::cout << "Using host buffers, KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=OFF" << std::endl;
    auto* a_buffer_0 = a_buffer_0_host_mirror.data();
    auto* a_buffer_1 = a_buffer_1_host_mirror.data();
#endif

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
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       a_buffer_0,
                       nullptr, // &mask_LIM[0][0],
                       a_buffer_0,
                       a_buffer_0) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Create() failed\n";
        return -1;
    }

    std::cout << an_operator << "\n";

    NaiveBenchmark([&](auto) {
        if(::koliop_Collision(an_operator,
                              dt,
                              a_buffer_1) != KOLIOP_STATUS_SUCCESS) {
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
