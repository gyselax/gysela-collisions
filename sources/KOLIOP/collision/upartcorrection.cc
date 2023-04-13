// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View6D,
                  typename View4D>
        static inline void
        DoComputeUparTCorrection(const Operator& an_operator,
                                 View6D&&        f5d_s,
                                 View4D&&        U_par_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& hat_As         = an_operator.hat_As_;
            const auto& mug            = an_operator.mug_;
            const auto& vparg          = an_operator.vparg_;
            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& B_norm         = an_operator.B_norm_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;
            const auto& T_s            = an_operator.T_s_;
            const auto& nu_ab          = an_operator.nu_ab_;
            const auto& q_par_on_nT_s  = an_operator.q_par_on_nT_s_;

            const auto  pSpecies_id            = an_operator.pSpecies_id_;
            const auto  collision_interspecies = an_operator.collision_interspecies_;
            const Flo64 dt                     = an_operator.dt_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 Bij            = B_norm(ir, itheta);
                const Flo64 na             = n_s(ir, itheta, iphi, pSpecies_id);
                Flo64       impulsion_coll = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jacob_vel = koliop::portability::MathematicalConstant<Flo64>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id);
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        impulsion_coll += coeff_intdvpar(ivpar) *
                                          coeff_intdmu(imu) *
                                          jacob_vel *
                                          vparg(ivpar) *
                                          f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                    }
                }

                const Flo64 U_par_coll = impulsion_coll / na;
                Flo64       T_coll     = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jacob_vel = koliop::portability::MathematicalConstant<Flo64>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id) / na;
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        T_coll += coeff_intdvpar(ivpar) *
                                  coeff_intdmu(imu) *
                                  (jacob_vel / 3.0) *
                                  ((vparg(ivpar) - U_par_coll) * (vparg(ivpar) - U_par_coll) +
                                   2 * mug(imu) * Bij) *
                                  f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                    }
                }

                const Flo64 hat_Aa      = (pSpecies_id);
                const Flo64 Vpara       = U_par_s(ir, itheta, iphi, pSpecies_id);
                Flo64       U_par_new   = U_par_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 q_par_on_nT = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 T_new       = T_s(ir, itheta, iphi, pSpecies_id);

                for(uint64_t ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    if((ispecies != pSpecies_id) && collision_interspecies) {
                        const Flo64 hat_Ab         = hat_As(ispecies);
                        const Flo64 Rat            = Kokkos::sqrt(hat_Aa / hat_Ab);
                        const Flo64 Vparb          = U_par_s(ir, itheta, iphi, ispecies);
                        const Flo64 q_parb_on_nbTb = q_par_on_nT_s(ir, itheta, iphi, ispecies);

                        const Flo64 k_ab = 1.0 / (1.0 +
                                                  hat_Aa * T_s(ir, itheta, iphi, ispecies) /
                                                      (hat_Ab * T_s(ir, itheta, iphi, pSpecies_id)));
                        const Flo64 k_ba = 1.0 / (1.0 +
                                                  hat_Ab * T_s(ir, itheta, iphi, pSpecies_id) /
                                                      (hat_Aa * T_s(ir, itheta, iphi, ispecies)));

                        const Flo64 nu = nu_ab(ir, itheta, iphi, pSpecies_id, ispecies);

                        const Flo64 R_ab = -nu * (Vpara - Rat * Vparb -
                                                  0.6 * k_ab * q_par_on_nT +
                                                  Rat * 0.6 * k_ba * q_parb_on_nbTb);
                        U_par_new += dt * R_ab;
                    }
                }

                const Flo64 coeff_Max_coll = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                                   T_coll *
                                                   Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * T_coll));

                const Flo64 coeff_Max_new = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                                  T_new *
                                                  Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * T_new));

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 u_coll = mug(imu) * Bij / T_coll;
                        const Flo64 u_new  = mug(imu) * Bij / T_new;

                        const Flo64 FMa_coll = coeff_Max_coll * Kokkos::exp(-u_coll -
                                                                            (vparg(ivpar) - U_par_coll) *
                                                                                (vparg(ivpar) - U_par_coll) / (2 * T_coll));
                        const Flo64 FMa_new  = coeff_Max_new * Kokkos::exp(-u_new -
                                                                           (vparg(ivpar) - U_par_new) *
                                                                               (vparg(ivpar) - U_par_new) / (2 * T_new));

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) += FMa_new - FMa_coll;
                    }
                }

                U_par_s(ir, itheta, iphi, pSpecies_id) = U_par_new;
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeUparTCorrection",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeUparTCorrection(Flo64* f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoComputeUparTCorrection(*this, a_device_visible_f5d_s, U_par_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s,
                                           koliop::utility::View::UHL6<Flo64>{f5d_s,
                                                                              the_mu_extent_, the_vpar_extent_,
                                                                              the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                              the_species_extent_});

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
