#include <KOLIOP/collisions.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.h"

int Test1() {
    static constexpr int8_t collision_interspecies = true;
    static constexpr double dt                     = 50.0;

    static constexpr size_t mu_extent      = 64;
    static constexpr size_t vpar_extent    = 128;
    static constexpr size_t r_extent       = 64;
    static constexpr size_t theta_extent   = 128;
    static constexpr size_t phi_extent     = 1;
    static constexpr size_t species_extent = 1;

    const size_t buffer_size = mu_extent * vpar_extent * theta_extent * r_extent * phi_extent * species_extent;

    Kokkos::View<double*> a_buffer_0_device_view("device_buffer", buffer_size);
    const auto            a_buffer_0_host_mirror = Kokkos::create_mirror_view(a_buffer_0_device_view);

    for(size_t i = 0; i < buffer_size; ++i) {
        a_buffer_0_host_mirror(i) = 1.0;
    }

    Kokkos::deep_copy(a_buffer_0_device_view, a_buffer_0_host_mirror);

    std::cout << "Using device buffers." << std::endl;
    auto* a_buffer_0 = a_buffer_0_device_view.data();

    // NOTE: the cache locality effect implied by spoofing
    // all these array as a single ptr is not visible on GPU.
    koliop::utility::View::UDL1<const double> hat_As{a_buffer_0, species_extent};
    koliop::utility::View::UDL1<const double> hat_Zs{a_buffer_0, species_extent};
    koliop::utility::View::UDL1<const double> mug{a_buffer_0, mu_extent};
    koliop::utility::View::UDL1<const double> vparg{a_buffer_0, vpar_extent};
    koliop::utility::View::UDL1<const double> coeff_intdmu{a_buffer_0, mu_extent};
    koliop::utility::View::UDL1<const double> coeff_intdvpar{a_buffer_0, vpar_extent};
    koliop::utility::View::UDL1<const double> coeff_AD_r{a_buffer_0, r_extent};
    koliop::utility::View::UDL2<const double> B_norm{a_buffer_0, r_extent, theta_extent};
    koliop::utility::View::UDL4<const double> Bstar_s{a_buffer_0, vpar_extent, r_extent, theta_extent, species_extent};

    koliop::collisions::Operator an_operator{
        theta_extent,
        r_extent,
        phi_extent,
        collision_interspecies,
        hat_As,
        hat_Zs,
        mug,
        vparg,
        coeff_intdmu,
        coeff_intdvpar,
        coeff_AD_r,
        B_norm,
        Bstar_s};

    koliop::utility::View::UDL6<double> f5d_s{a_buffer_0, mu_extent, vpar_extent, theta_extent, r_extent, phi_extent, species_extent};

    an_operator.ComputeMoments(f5d_s);
    an_operator.PrecomputeNu();
    an_operator.ComputeMaxwellianEvolution(dt, f5d_s);
    an_operator.ComputeQparOnNT(f5d_s);
    an_operator.ComputeCpar(dt, f5d_s);
    an_operator.ComputeMaxwellianUnshiftedLoc(4, f5d_s);
    an_operator.Apply(dt, f5d_s);
    an_operator.FenceAll();
    an_operator.ComputeDensityCorrection(f5d_s);
    an_operator.ComputeUparTCorrection(dt, f5d_s);
    an_operator.ExtractRaw(koliop::collisions::Operator::Diagnostics::k_n_s, a_buffer_0);
    an_operator.FenceAll();
    return 0;
}

int Test2() {
    static constexpr int8_t collision_interspecies = true;
    static constexpr double dt                     = 50.0;

    static constexpr size_t mu_extent      = 64;
    static constexpr size_t vpar_extent    = 128;
    static constexpr size_t r_extent       = 64;
    static constexpr size_t theta_extent   = 128;
    static constexpr size_t phi_extent     = 32;
    static constexpr size_t species_extent = 2;

    const size_t buffer_size = mu_extent * vpar_extent * theta_extent * r_extent * phi_extent * species_extent;

    Kokkos::View<double*> a_buffer_0_device_view("device_buffer", buffer_size);
    const auto            a_buffer_0_host_mirror = Kokkos::create_mirror_view(a_buffer_0_device_view);

    for(size_t i = 0; i < buffer_size; ++i) {
        a_buffer_0_host_mirror(i) = 1.0;
    }

    Kokkos::deep_copy(a_buffer_0_device_view, a_buffer_0_host_mirror);

    auto* a_buffer_0 = a_buffer_0_device_view.data();

    koliop::utility::View::UDL1<const double> hat_As{a_buffer_0, species_extent};
    koliop::utility::View::UDL1<const double> hat_Zs{a_buffer_0, species_extent};
    koliop::utility::View::UDL1<const double> mug{a_buffer_0, mu_extent};
    koliop::utility::View::UDL1<const double> vparg{a_buffer_0, vpar_extent};
    koliop::utility::View::UDL1<const double> coeff_intdmu{a_buffer_0, mu_extent};
    koliop::utility::View::UDL1<const double> coeff_intdvpar{a_buffer_0, vpar_extent};
    koliop::utility::View::UDL1<const double> coeff_AD_r{a_buffer_0, r_extent};
    koliop::utility::View::UDL2<const double> B_norm{a_buffer_0, r_extent, theta_extent};
    koliop::utility::View::UDL4<const double> Bstar_s{a_buffer_0, vpar_extent, r_extent, theta_extent, species_extent};

    koliop::collisions::Operator an_operator{
        theta_extent,
        r_extent,
        phi_extent,
        collision_interspecies,
        hat_As,
        hat_Zs,
        mug,
        vparg,
        coeff_intdmu,
        coeff_intdvpar,
        coeff_AD_r,
        B_norm,
        Bstar_s};

    koliop::utility::View::UDL6<double> f5d_s{a_buffer_0, mu_extent, vpar_extent, theta_extent, r_extent, phi_extent, species_extent};

    NaiveBenchmark([&](auto) {
        an_operator.Collisions(dt, f5d_s);
        an_operator.FenceAll();
    });

    return 0;
}

int main(int, char**) {
    Initialize(8, 0, true);
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
    Deinitialize();
    std::cout << "Deinitialized()\n";
    return 0;
}
