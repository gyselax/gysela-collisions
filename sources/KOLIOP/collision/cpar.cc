// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View6D>
        static inline void
        DoComputeCpar(const Operator& an_operator,
                      View6D&&        f5d_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& hat_As        = an_operator.hat_As_;
            const auto& hat_Zs        = an_operator.hat_Zs_;
            const auto& mug           = an_operator.mug_;
            const auto& vparg         = an_operator.vparg_;
            const auto& rg            = an_operator.rg_;
            const auto& nustar0_r     = an_operator.nustar0_r_;
            const auto& q_r           = an_operator.q_r_;
            const auto& B_norm        = an_operator.B_norm_;
            const auto& n_s           = an_operator.n_s_;
            const auto& U_par_s       = an_operator.U_par_s_;
            const auto& T_s           = an_operator.T_s_;
            const auto& q_par_on_nT_s = an_operator.q_par_on_nT_s_;

            const auto&            pSpecies_id = an_operator.pSpecies_id_;
            const ComputeFloatType R0_inverse  = 1.0 / an_operator.R0_;
            const ComputeFloatType dt          = an_operator.dt_ / 2.0;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): Assuming materialization, we could flatten
                // the loop even more, up to Nmu_.
                // TODO(Etienne M): We could inter exchange some loops ?
                // TODO(Etienne M): There is quite a lot to simplify here.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const ComputeFloatType hat_Aa     = hat_As(pSpecies_id);
                const ComputeFloatType hat_Za     = hat_Zs(pSpecies_id);
                const ComputeFloatType inv_hat_Aa = 1.0 / hat_Aa;

                const ComputeFloatType smcsfqri = q_r(ir);
                const ComputeFloatType Bij      = B_norm(ir, itheta);

                const ComputeFloatType na          = n_s(ir, itheta, iphi, pSpecies_id);
                const ComputeFloatType Vpara       = U_par_s(ir, itheta, iphi, pSpecies_id);
                const ComputeFloatType Ta          = T_s(ir, itheta, iphi, pSpecies_id);
                const ComputeFloatType q_par_on_nT = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);

                const ComputeFloatType nustaR0_i = nustar0_r(ir);
                const ComputeFloatType epstemp   = rg(ir) * R0_inverse;
                const ComputeFloatType vTa       = Kokkos::sqrt(Ta / hat_Aa);
                const ComputeFloatType coeff1    = 1.0 / (2.0 * Ta);
                const ComputeFloatType coeff3    = na / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));
                const ComputeFloatType AD        = nustaR0_i * koliop::portability::MathematicalConstant<ComputeFloatType>::kSqrt2 * epstemp * Kokkos::sqrt(epstemp) / ((1.0 / R0_inverse) * (smcsfqri != koliop::portability::GyselaConstant<Flo64>::kInfinity ? smcsfqri : 1.0));

                for(OrdinalType ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    const ComputeFloatType hat_Ab         = hat_As(ispecies);
                    const ComputeFloatType inv_hat_Ab     = 1.0 / hat_Ab;
                    const ComputeFloatType hat_Zb         = hat_Zs(ispecies);
                    const ComputeFloatType nb             = n_s(ir, itheta, iphi, ispecies);
                    const ComputeFloatType Vparb          = U_par_s(ir, itheta, iphi, ispecies);
                    const ComputeFloatType Tb             = T_s(ir, itheta, iphi, ispecies);
                    const ComputeFloatType q_parb_on_nbTb = q_par_on_nT_s(ir, itheta, iphi, ispecies);

                    const ComputeFloatType Rat    = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const ComputeFloatType vTb    = Kokkos::sqrt(Tb / hat_Ab);
                    const ComputeFloatType coeff2 = vTa / vTb;
                    const ComputeFloatType coeff4 = (3.0 / 5.0) * Rat * q_parb_on_nbTb / (1.0 + (Ta * hat_Ab) / (Tb * hat_Aa));

                    for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                        for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            const ComputeFloatType coeff5         = (vparg(ivpar) / Ta);
                            const ComputeFloatType xa             = Kokkos::sqrt(coeff1 * (2.0 * mug(imu) * Bij + vparg(ivpar) * vparg(ivpar)));
                            const ComputeFloatType xb             = xa * coeff2;
                            const ComputeFloatType FM0a           = coeff3 * Kokkos::exp(-xa * xa);
                            const ComputeFloatType Theta_func_tmp = xb > 1E-15 ?
                                                                        koliop::portability::GyselaConstant<ComputeFloatType>::kConst_theta * ((Kokkos::erf(xb) - xb * (koliop::portability::GyselaConstant<Flo64>::kCst_phiprime * exp(-xb * xb))) / (2.0 * xb * xb)) / xb :
                                                                        1.0;
                            const ComputeFloatType nu_sab         = (AD * hat_Za * hat_Za * hat_Zb * hat_Zb * nb) * inv_hat_Aa * (inv_hat_Aa + inv_hat_Ab) * Theta_func_tmp / (vTb * vTb * vTb);

                            f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) += dt * FM0a * nu_sab * coeff5 * (Rat * Vparb - Vpara + q_par_on_nT * (1.0 - 2.0 * xa * xa * 0.2) - coeff4);
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeCpar",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeCpar(Flo64* f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoComputeCpar(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s,
                                           koliop::utility::View::UHL6<Flo64>{f5d_s,
                                                                              static_cast<std::size_t>(the_mu_extent_), static_cast<std::size_t>(the_vpar_extent_),
                                                                              static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                              static_cast<std::size_t>(the_species_extent_)});

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
