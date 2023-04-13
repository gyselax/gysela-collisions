// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename Array6D,
                  typename Array4D>
        static inline void
        DoComputeMoments(const Operator& an_operator,
                         Array6D&&       f5d_s,
                         Array4D&&       n_s,
                         Array4D&&       U_par_s,
                         Array4D&&       T_s) {

            const uint64_t& the_mu_extent      = an_operator.the_mu_extent_;
            const uint64_t& the_vpar_extent    = an_operator.the_vpar_extent_;
            const uint64_t& the_r_extent       = an_operator.the_r_extent_;
            const uint64_t& the_theta_extent   = an_operator.the_theta_extent_;
            const uint64_t& the_phi_extent     = an_operator.the_phi_extent_;
            const uint64_t& the_species_extent = an_operator.the_species_extent_;

            const auto& mug            = an_operator.mug_;
            const auto& vparg          = an_operator.vparg_;
            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& B_norm         = an_operator.B_norm_;
            const auto& Bstar_s        = an_operator.Bstar_s_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi, auto ispecies) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                Flo64 na    = 0.0;
                Flo64 vpara = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 f5d_s_value = Kokkos::max(1.0E-40,
                                                              f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const Flo64 ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                           (coeff_intdmu(imu) * f5d_s_value);
                        na += ftmp;
                        vpara += vparg(ivpar) * ftmp;

                        f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) = f5d_s_value;
                    }
                }

                vpara = vpara / na;

                Flo64 Ta = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        // TODO(Etienne M): Avoid this recomputation ? On
                        // Gysela fortran, a temporary buffer is used but
                        // that would hardly fit on a GPU.
                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const Flo64 ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                           (coeff_intdmu(imu) * f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));
                        Ta += ((vparg(ivpar) - vpara) * (vparg(ivpar) - vpara) +
                               2.0 * mug(imu) * B_norm(ir, itheta)) *
                              ftmp;
                    }
                }

                Ta = Ta / (3.0 * na);
                na = koliop::portability::MathematicalConstant<Flo64>::k2Pi * na;

                n_s(ir, itheta, iphi, ispecies)     = na;
                U_par_s(ir, itheta, iphi, ispecies) = vpara;
                T_s(ir, itheta, iphi, ispecies)     = Ta;
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent,
                                                                                              the_species_extent}};

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeMoments(Flo64* f5d_s) {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = koliop::utility::View::MDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                                             the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = koliop::utility::View::UHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                                           f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeMoments(*this, a_device_visible_f5d_s, n_s_, U_par_s_, T_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
