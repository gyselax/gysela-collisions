#include <KOLIOP/collisions.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.h"

int Test2() {
    static constexpr int8_t collision_interspecies = true;
    static constexpr int8_t SOL                    = false;
    static constexpr int8_t LIM                    = false;
    static constexpr double dt                     = 50.0;

    static constexpr size_t mu_extent   = 64;
    static constexpr size_t vpar_extent = 128;

    static constexpr size_t r_extent       = 16;
    static constexpr size_t theta_extent   = 8;
    static constexpr size_t phi_extent     = 4;
    static constexpr size_t species_extent = 2;

    static constexpr size_t global_domain_r_extent     = r_extent;
    static constexpr size_t global_domain_theta_extent = theta_extent + 1;
    static constexpr size_t global_domain_r_offset     = 0;
    static constexpr size_t global_domain_theta_offset = 0;

    static constexpr int64_t ir_SOL_separatrix = global_domain_r_extent;

    const size_t buffer_size = mu_extent * vpar_extent * theta_extent * r_extent * phi_extent * species_extent;

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

    auto* a_buffer_0 = a_buffer_0_device_view.data();
    auto* a_buffer_1 = a_buffer_1_device_view.data();

    koliop::utility::View::UDL1<const double> hat_As{a_buffer_0, species_extent};
    koliop::utility::View::UDL1<const double> hat_Zs{a_buffer_0, species_extent};
    koliop::utility::View::UDL1<const double> mug{a_buffer_0, mu_extent};
    koliop::utility::View::UDL1<const double> vparg{a_buffer_0, vpar_extent};
    koliop::utility::View::UDL1<const double> coeff_intdmu{a_buffer_0, mu_extent};
    koliop::utility::View::UDL1<const double> coeff_intdvpar{a_buffer_0, vpar_extent};
    koliop::utility::View::UDL1<const double> coeff_AD_r{a_buffer_0, r_extent};
    koliop::utility::View::UDL1<const double> mask_buffer_r{a_buffer_0, global_domain_r_extent};
    koliop::utility::View::UDL2<const double> mask_LIM{a_buffer_0, global_domain_r_extent, global_domain_theta_extent};
    koliop::utility::View::UDL2<const double> B_norm{a_buffer_0, global_domain_r_extent, global_domain_theta_extent};
    koliop::utility::View::UDL4<const double> Bstar_s{a_buffer_0, vpar_extent, r_extent, theta_extent, species_extent};

    koliop::collisions::Operator an_operator{
        mu_extent,
        vpar_extent,
        theta_extent,
        r_extent,
        phi_extent,
        species_extent,
        global_domain_r_extent,
        global_domain_theta_extent,
        global_domain_r_offset,
        global_domain_theta_offset,
        collision_interspecies,
        SOL,
        LIM,
        ir_SOL_separatrix,
        hat_As,
        hat_Zs,
        mug,
        vparg,
        coeff_intdmu,
        coeff_intdvpar,
        coeff_AD_r,
        mask_buffer_r,
        mask_LIM,
        B_norm,
        Bstar_s};

    koliop::utility::View::UDL6<double> f5d_s{a_buffer_1, mu_extent, vpar_extent, theta_extent, r_extent, phi_extent, species_extent};

    NaiveBenchmark([&](auto) {
        an_operator.Collisions(dt, f5d_s);
        an_operator.FenceAll();
    });

    return 0;
}

int main(int, char**) {
    Initialize(8, 0, true);
    if(Test2() < 0) {
        return -1;
    }
    Deinitialize();
    std::cout << "Deinitialized()\n";
    return 0;
}
