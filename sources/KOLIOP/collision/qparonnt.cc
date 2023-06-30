// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View6D,
                  typename View4D>
        static inline void
        DoComputeQparOnNT(const Operator& an_operator,
                          const View6D&   f5d_s,
                          View4D&&        q_par_on_nT_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

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

                const ComputeFloatType Bij   = B_norm(ir, itheta);
                const ComputeFloatType na    = n_s(ir, itheta, iphi, ispecies);
                const ComputeFloatType Vpara = U_par_s(ir, itheta, iphi, ispecies);
                const ComputeFloatType Ta    = T_s(ir, itheta, iphi, ispecies);
                // TODO(Etienne M): SIMPLIFY as std::pow(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta, -3 / 2);
                const ComputeFloatType coeff = 1.0 / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta));

                ComputeFloatType q_par_on_nT_s_reduction = 0.0;

                for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const ComputeFloatType jac = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                 Bstar_s(ivpar, ir, itheta, ispecies);

                    const ComputeFloatType vparg_minus_vpara         = vparg(ivpar) - Vpara;
                    const ComputeFloatType vparg_minus_vpara_squared = vparg_minus_vpara * vparg_minus_vpara;
                    const ComputeFloatType vp2                       = vparg_minus_vpara_squared / (2.0 * Ta);

                    for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                        const ComputeFloatType u_tmp   = mug(imu) * Bij / Ta;
                        const ComputeFloatType FMa_tmp = na * Kokkos::exp(-u_tmp - vp2) * coeff;
                        const ComputeFloatType v2      = 2.0 * mug(imu) * Bij + vparg_minus_vpara_squared;
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

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<const Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoComputeQparOnNT(*this, a_device_visible_f5d_s, q_par_on_nT_s_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
