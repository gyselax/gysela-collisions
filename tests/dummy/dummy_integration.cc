#include <KOLIOP/koliop.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

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

    static constexpr size_t the_mu_extent      = 64;
    static constexpr size_t the_vpar_extent    = 128;
    static constexpr size_t the_r_extent       = 64;
    static constexpr size_t the_theta_extent   = 128;
    static constexpr size_t the_phi_extent     = 1;
    static constexpr size_t the_species_extent = 1;

    static constexpr size_t the_global_domain_r_extent     = the_r_extent;
    static constexpr size_t the_global_domain_theta_extent = the_theta_extent + 1;
    static constexpr size_t the_global_domain_r_offset     = 0;
    static constexpr size_t the_global_domain_theta_offset = 0;

    std::vector<double> a_big_dummy_array{};
    a_big_dummy_array.resize(the_mu_extent *
                                 the_vpar_extent *
                                 the_r_extent *
                                 the_theta_extent *
                                 the_phi_extent *
                                 the_species_extent,
                             1.0);

    std::vector<double> a_big_dummy_f5d_s{};
    a_big_dummy_f5d_s.resize(the_mu_extent *
                                 the_vpar_extent *
                                 the_r_extent *
                                 the_theta_extent *
                                 the_phi_extent *
                                 the_species_extent,
                             1.0);

    if(::koliop_Create(&an_operator,
                       the_mu_extent,
                       the_vpar_extent,
                       the_r_extent,
                       the_theta_extent,
                       the_phi_extent,
                       the_species_extent,
                       the_global_domain_r_extent,
                       the_global_domain_theta_extent,
                       the_global_domain_r_offset,
                       the_global_domain_theta_offset,
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

    double n_s[the_species_extent][the_phi_extent][the_theta_extent][the_r_extent]{{{{42.0}}}};

    ::koliop_ExtractRaw(an_operator,
                        KOLIOP_EXTRACTION_IDENTIFIER_n_s,
                        &n_s[0][0][0]);

    ::koliop_Fence(an_operator);

    if(::koliop_Release(an_operator) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Release() failed\n";
    }

    return 0;
}

int Test2() {
    // Takes 1.89 sec on MI250X.
    //       0.78 sec on MI300A.

    // NOTE: This function is used to test the GPU code. We pass the operator
    // dummy arrays that get copied to the GPU anyway (assuming
    // koliop_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE=OFF) thus being
    // disjoint (avoiding cache effects).

    ::koliop_Operator an_operator;

    static constexpr int8_t  collision_interspecies = true;
    static constexpr int8_t  SOL                    = false;
    static constexpr int8_t  LIM                    = false;
    static constexpr double  dt                     = 50.0;

    static constexpr size_t the_mu_extent      = 64;
    static constexpr size_t the_vpar_extent    = 128;
    static constexpr size_t the_r_extent       = 64;
    static constexpr size_t the_theta_extent   = 128;
    static constexpr size_t the_phi_extent     = 32;
    static constexpr size_t the_species_extent = 2;

    static constexpr size_t the_global_domain_r_extent     = the_r_extent;
    static constexpr size_t the_global_domain_theta_extent = the_theta_extent + 1;
    static constexpr size_t the_global_domain_r_offset     = 0;
    static constexpr size_t the_global_domain_theta_offset = 0;

    static constexpr int64_t ir_SOL_separatrix = the_global_domain_r_extent;

    std::vector<double> a_big_dummy_array{};
    a_big_dummy_array.resize(the_mu_extent *
                                 the_vpar_extent *
                                 the_r_extent *
                                 the_theta_extent *
                                 the_phi_extent *
                                 the_species_extent,
                             1.0);

    std::vector<double> a_big_dummy_f5d_s{};
    a_big_dummy_f5d_s.resize(the_mu_extent *
                                 the_vpar_extent *
                                 the_r_extent *
                                 the_theta_extent *
                                 the_phi_extent *
                                 the_species_extent,
                             1.0);

    if(::koliop_Create(&an_operator,
                       the_mu_extent,
                       the_vpar_extent,
                       the_r_extent,
                       the_theta_extent,
                       the_phi_extent,
                       the_species_extent,
                       the_global_domain_r_extent,
                       the_global_domain_theta_extent,
                       the_global_domain_r_offset,
                       the_global_domain_theta_offset,
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

    if(::koliop_Collision(an_operator,
                          dt,
                          a_big_dummy_f5d_s.data()) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Collision() failed\n";
        return -1;
    }

    for(int i = 0; i < 10; ++i) {
        const auto time_point = std::chrono::steady_clock::now();

        if(::koliop_Collision(an_operator,
                              dt,
                              a_big_dummy_f5d_s.data()) != KOLIOP_STATUS_SUCCESS) {
            std::cout << "::koliop_Collision() failed\n";
            return -1;
        }

        const auto duration = std::chrono::steady_clock::now() - time_point;
        std::cout << (duration.count() / 1'000'000) << " ms\n" << std::flush;
    }

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
