// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {

        template <typename Array6D>
        static inline void
        DoComputeDensityCorrection(const State& a_state,
                                   Array6D&&    f5d_s) {

            const auto& the_mu_extent    = a_state.the_mu_extent_;
            const auto& the_vpar_extent  = a_state.the_vpar_extent_;
            const auto& the_r_extent     = a_state.the_r_extent_;
            const auto& the_theta_extent = a_state.the_theta_extent_;
            const auto& the_phi_extent   = a_state.the_phi_extent_;

            const auto& coeff_intdmu   = a_state.coeff_intdmu_;
            const auto& coeff_intdvpar = a_state.coeff_intdvpar_;
            const auto& Bstar_s        = a_state.Bstar_s_;
            const auto& n_s            = a_state.n_s_;

            const auto pSpecies_id = a_state.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                Flo64 na = 0.0;
                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jacob_vel = koliop::portability::MathematicalConstant<Flo64>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id);
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 ftmp = Kokkos::max(f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id),
                                                       0.0);

                        na += jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp;

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = ftmp;
                    }
                }

                const Flo64 na_inverse = 1.0 / na;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) *= n_s(ir, itheta, iphi, pSpecies_id) * na_inverse;
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeDensityCorrection",
                                 a_policy, the_kernel);
        }

        void State::ComputeDensityCorrection(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeDensityCorrection(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
