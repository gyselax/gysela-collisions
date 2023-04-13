// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {

        template <typename Array6D>
        static inline void
        DoComputeCpar(const Operator& an_operator,
                      Array6D&&       f5d_s) {

            const uint64_t& the_mu_extent      = an_operator.the_mu_extent_;
            const uint64_t& the_vpar_extent    = an_operator.the_vpar_extent_;
            const uint64_t& the_r_extent       = an_operator.the_r_extent_;
            const uint64_t& the_theta_extent   = an_operator.the_theta_extent_;
            const uint64_t& the_phi_extent     = an_operator.the_phi_extent_;
            const uint64_t& the_species_extent = an_operator.the_species_extent_;

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

            const uint64_t& pSpecies_id = an_operator.pSpecies_id_;
            const Flo64     R0_inverse  = 1.0 / an_operator.R0_;
            const Flo64     dt          = an_operator.dt_ / 2.0;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): Assuming materialization, we could flatten
                // the loop even more, up to Nmu_.
                // TODO(Etienne M): We could inter exchange some loops ?
                // TODO(Etienne M): There is quite a lot to simplify here.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 hat_Aa     = hat_As(pSpecies_id);
                const Flo64 hat_Za     = hat_Zs(pSpecies_id);
                const Flo64 inv_hat_Aa = 1.0 / hat_Aa;

                const Flo64 smcsfqri = q_r(ir);
                const Flo64 Bij      = B_norm(ir, itheta);

                const Flo64 na          = n_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 Vpara       = U_par_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 Ta          = T_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 q_par_on_nT = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);

                const Flo64 nustaR0_i = nustar0_r(ir);
                const Flo64 epstemp   = rg(ir) * R0_inverse;
                const Flo64 vTa       = Kokkos::sqrt(Ta / hat_Aa);
                const Flo64 coeff1    = 1.0 / (2.0 * Ta);
                const Flo64 coeff3    = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));
                const Flo64 AD        = nustaR0_i * koliop::portability::MathematicalConstant<Flo64>::kSQRT2 * epstemp * Kokkos::sqrt(epstemp) / ((1.0 / R0_inverse) * (smcsfqri != koliop::portability::GyselaConstant<Flo64>::kInfinity ? smcsfqri : 1.0));

                for(uint64_t ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    const Flo64 hat_Ab         = hat_As(ispecies);
                    const Flo64 inv_hat_Ab     = 1.0 / hat_Ab;
                    const Flo64 hat_Zb         = hat_Zs(ispecies);
                    const Flo64 nb             = n_s(ir, itheta, iphi, ispecies);
                    const Flo64 Vparb          = U_par_s(ir, itheta, iphi, ispecies);
                    const Flo64 Tb             = T_s(ir, itheta, iphi, ispecies);
                    const Flo64 q_parb_on_nbTb = q_par_on_nT_s(ir, itheta, iphi, ispecies);

                    const Flo64 Rat    = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const Flo64 vTb    = Kokkos::sqrt(Tb / hat_Ab);
                    const Flo64 coeff2 = vTa / vTb;
                    const Flo64 coeff4 = (3.0 / 5.0) * Rat * q_parb_on_nbTb / (1.0 + (Ta * hat_Ab) / (Tb * hat_Aa));

                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            const Flo64 coeff5         = (vparg(ivpar) / Ta);
                            const Flo64 xa             = Kokkos::sqrt(coeff1 * (2.0 * mug(imu) * Bij + vparg(ivpar) * vparg(ivpar)));
                            const Flo64 xb             = xa * coeff2;
                            const Flo64 FM0a           = coeff3 * Kokkos::exp(-xa * xa);
                            const Flo64 Theta_func_tmp = xb > 1E-15 ?
                                                             koliop::portability::GyselaConstant<Flo64>::kConst_theta * ((Kokkos::erf(xb) - xb * (koliop::portability::GyselaConstant<Flo64>::kCst_phiprime * exp(-xb * xb))) / (2.0 * xb * xb)) / xb :
                                                             1.0;
                            const Flo64 nu_sab         = (AD * hat_Za * hat_Za * hat_Zb * hat_Zb * nb) * inv_hat_Aa * (inv_hat_Aa + inv_hat_Ab) * Theta_func_tmp / (vTb * vTb * vTb);

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
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeCpar(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
