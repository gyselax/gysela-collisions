// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename Array6D,
                  typename Array4D>
        static inline void
        DoComputeQparOnNT(const Operator& an_operator,
                          const Array6D&  f5d_s,
                          Array4D&&       q_par_on_nT_s) {

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
            const auto& n_s            = an_operator.n_s_;
            const auto& U_par_s        = an_operator.U_par_s_;
            const auto& T_s            = an_operator.T_s_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi, auto ispecies) {
                // TODO(Etienne M): Should we flatten even more the loops ?
                // Potentially computing/loading redundant values (Bij, na etc.)
                // but extracting more parallelisms. The thing is that ivpar and
                // imu tends to be larger than ir, itheta or iphi.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 Bij   = B_norm(ir, itheta);
                const Flo64 na    = n_s(ir, itheta, iphi, ispecies);
                const Flo64 Vpara = U_par_s(ir, itheta, iphi, ispecies);
                const Flo64 Ta    = T_s(ir, itheta, iphi, ispecies);
                // TODO(Etienne M): SIMPLIFY as std::pow(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta, -3 / 2);
                const Flo64 coeff = 1.0 / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));

                Flo64 q_par_on_nT_s_reduction = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jac = koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                      Bstar_s(ivpar, ir, itheta, ispecies);

                    const Flo64 vparg_minus_vpara         = vparg(ivpar) - Vpara;
                    const Flo64 vparg_minus_vpara_squared = vparg_minus_vpara * vparg_minus_vpara;
                    const Flo64 vp2                       = vparg_minus_vpara_squared / (2.0 * Ta);

                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 u_tmp   = mug(imu) * Bij / Ta;
                        const Flo64 FMa_tmp = na * Kokkos::exp(-u_tmp - vp2) * coeff;
                        const Flo64 v2      = 2.0 * mug(imu) * Bij + vparg_minus_vpara_squared;
                        q_par_on_nT_s_reduction += 0.5 *
                                                   jac *
                                                   coeff_intdmu(imu) *
                                                   coeff_intdvpar(ivpar) *
                                                   vparg_minus_vpara *
                                                   v2 *
                                                   (f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) - FMa_tmp);
                    }
                }

                q_par_on_nT_s(ir, itheta, iphi, ispecies) = q_par_on_nT_s_reduction / (na * Ta);
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent,
                                                                                              the_species_extent}};

            Kokkos::parallel_for("DoComputeQparOnNT",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeQparOnNT(const Flo64* f5d_s) {
            // TODO(Etienne M): COPY redundant.

            // TODO(Etienne M): On host we want the view to have the same type
            // (operator= is used.). On GPU, we need a deep copy and
            // a_device_visible_f5d_s can not be const..
            auto a_device_visible_f5d_s = koliop::utility::View::MDL6</* const */ Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                                                         the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = koliop::utility::View::UHL6<const Flo64>{// "unmanaged_host_f5d_s",
                                                                                 f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeQparOnNT(*this, a_device_visible_f5d_s, q_par_on_nT_s_);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
