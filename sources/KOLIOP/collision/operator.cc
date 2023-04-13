// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View3D,
                  typename MirrorView2D>
        static inline void
        DoPrepareCjl(const MirrorView2D& a_host_visible_C_j,
                     View3D&             C_jl) {

            // a_host_visible_C_jl is zero initialized.
            auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl);

            const auto CoreLoop = [&](auto j_tmp, auto l_tmp) {
                for(uint64_t i = 0; i <= (j_tmp + l_tmp); ++i) {
                    // This seems unnecessary ?
                    a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                }

                for(uint64_t alpha = 0; alpha <= j_tmp; ++alpha) {
                    for(uint64_t beta = 0; beta <= l_tmp; ++beta) {
                        a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) += a_host_visible_C_j(alpha, j_tmp) *
                                                                           a_host_visible_C_j(beta, l_tmp);
                    }
                }
            };

            for(uint64_t i = 0; i < Operator::kMaximumLaguerrePolynomialDegree; ++i) {
                for(uint64_t j = 0; j < Operator::kMaximumLaguerrePolynomialDegree; ++j) {
                    CoreLoop(i, j);
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_jl, C_jl);
        }

        template <typename View3D,
                  typename View2D>
        static inline void
        DoPrepareCj_Cjl(View2D& C_j,
                        View3D& C_jl) {

            // TODO(Etienne M): This small matrix is know at compile time !

            // a_host_visible_C_j is zero initialized.
            auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

            for(uint64_t j = 0; j < Operator::kMaximumLaguerrePolynomialDegree; ++j) {
                a_host_visible_C_j(0, j) = 1.0;
                a_host_visible_C_j(1, j) = static_cast<Flo64>(-static_cast<int64_t>(j));
            }

            if(Operator::kMaximumLaguerrePolynomialDegree > 2) {
                a_host_visible_C_j(2, 2) = 0.5;
                if(Operator::kMaximumLaguerrePolynomialDegree > 3) {
                    // NOTE: Assuming kMaximumLaguerrePolynomialDegree = 3, this
                    // is never evaluated.
                    a_host_visible_C_j(2, 3) = 1.5;
                    a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_j, C_j);

            DoPrepareCjl(a_host_visible_C_j, C_jl);
        }

        static inline void
        DoPrepareNuStar(const Operator&                                 an_operator,
                        koliop::utility::View::MDD3<Flo64>&             nustar_coeff,
                        const koliop::utility::View::UHL1<const Flo64>& a_host_visible_hat_As,
                        const koliop::utility::View::UHL1<const Flo64>& a_host_visible_hat_Zs,
                        const koliop::utility::View::UHL1<const Flo64>& a_host_visible_rg,
                        const koliop::utility::View::UHL1<const Flo64>& a_host_visible_nustar0_r,
                        const koliop::utility::View::UHL1<const Flo64>& a_host_visible_q_r,
                        const koliop::utility::View::UHL1<const Flo64>& a_host_visible_mask_buffer_r) {

            // TODO(Etienne M): I observe some slight numerical discrepancy
            // (1e-14 ~ ) when comparing the result of Gysela and koliop.

            koliop::utility::View::MHL1<Flo64> a_host_visible_mask_space{Kokkos::view_alloc(Kokkos::WithoutInitializing, "a_host_visible_mask_space"),
                                                                         an_operator.the_r_extent_};

            if(a_host_visible_q_r(0) < koliop::portability::GyselaConstant<Flo64>::kInfinity) {
                for(uint64_t i = 0; i < an_operator.the_r_extent_; ++i) {
                    a_host_visible_mask_space(i) = 1.0 / a_host_visible_q_r(i);
                }
            }

            if(!(an_operator.LIM_ || an_operator.SOL_)) {
                for(uint64_t i = 0; i < an_operator.the_r_extent_; ++i) {
                    a_host_visible_mask_space(i) *= (1.0 - a_host_visible_mask_buffer_r(i));
                }
            }

            const auto a_host_visible_nustar_coeff = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                            Kokkos::WithoutInitializing),
                                                                                         nustar_coeff);

            const auto R0_inverse = 1.0 / an_operator.R0_;

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
            for(uint64_t ispeciesB = 0; ispeciesB < an_operator.the_species_extent_; ++ispeciesB) {
                for(uint64_t ispeciesA = 0; ispeciesA < an_operator.the_species_extent_; ++ispeciesA) {
                    const Flo64 hat_Aa = a_host_visible_hat_As(ispeciesA);
                    const Flo64 hat_Za = a_host_visible_hat_Zs(ispeciesA);
                    const Flo64 hat_Ab = a_host_visible_hat_As(ispeciesB);
                    const Flo64 hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                    const Flo64 AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                            (1.0 / hat_Aa) *
                                            (1.0 / hat_Aa + 1.0 / hat_Ab);

                    for(uint64_t ir = 0; ir < an_operator.the_r_extent_; ++ir) {
                        // Computation of
                        // AD(r) = sqrt(2) * eps ^ (3 / 2) / (q R0_)*nustaR0_(r)
                        // which is independent on the species considered (eps = r/R0_)
                        const Flo64 eps = a_host_visible_rg(ir) * R0_inverse;

                        const Flo64 AD = a_host_visible_nustar0_r(ir) *
                                         a_host_visible_mask_space(ir) *
                                         // TODO(Etienne M): SIMPLIFY
                                         // sqrt(2)*(rg*R0_inverse)*sqrt(rg*R0_inverse)*R0_inverse_ =
                                         // sqrt(2)*pow(r, 5/2)*pow(rg, 3/2)
                                         // sqrt(2)*pow(r, 5/2) is constant here.
                                         koliop::portability::MathematicalConstant<Flo64>::kSQRT2 *
                                         eps *
                                         Kokkos::sqrt(eps) *
                                         R0_inverse;

                        // Computation of nu_ab =
                        // (AD * hat_Za ^ 2 * hat_Zb ^ 2 * Nb) /
                        // (vTa ^ 2 + vTb ^ 2) ^ (3 / 2) * (1 / hat_Aa) *
                        // (1 / hat_Aa + 1 / hat_Ab)
                        a_host_visible_nustar_coeff(ispeciesA, ispeciesB, ir) = AD * AD_factor;
                    }
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_nustar_coeff, nustar_coeff);
        }

        Operator::Operator(uint64_t                                        pSpecies_id,
                           uint64_t                                        mu_id,
                           uint64_t                                        the_mu_extent,
                           uint64_t                                        the_vpar_extent,
                           uint64_t                                        the_r_extent,
                           uint64_t                                        the_theta_extent,
                           uint64_t                                        the_phi_extent,
                           uint64_t                                        the_species_extent,
                           Flo64                                           R0,
                           bool                                            collision_interspecies,
                           bool                                            SOL,
                           bool                                            LIM,
                           uint64_t                                        ir_SOL_separatrix,
                           Flo64                                           dt,
                           const koliop::utility::View::UHL2<const Flo64>& a_host_visible_comb_mat,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_hat_As,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_hat_Zs,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_mug,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_vparg,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_coeff_intdmu,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_coeff_intdvpar,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_rg,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_nustar0_r,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_q_r,
                           const koliop::utility::View::UHL1<const Flo64>& a_host_visible_mask_buffer_r,
                           const koliop::utility::View::UHL2<const Flo64>& a_host_visible_mask_LIM,
                           const koliop::utility::View::UHL2<const Flo64>& a_host_visible_B_norm,
                           const koliop::utility::View::UHL4<const Flo64>& a_host_visible_Bstar_s)
            // Constant, externally computed values
            : pSpecies_id_{pSpecies_id}
            , mu_id_{mu_id}
            , the_mu_extent_{the_mu_extent}
            , the_vpar_extent_{the_vpar_extent}
            , the_r_extent_{the_r_extent}
            , the_theta_extent_{the_theta_extent}
            , the_phi_extent_{the_phi_extent}
            , the_species_extent_{the_species_extent}
            , R0_{R0}
            , collision_interspecies_{collision_interspecies}
            , SOL_{SOL}
            , LIM_{LIM}
            , ir_SOL_separatrix_{ir_SOL_separatrix}
            , dt_{dt}
            , comb_mat_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "comb_mat_"),
                        2 * kMaximumLaguerrePolynomialDegree, 2 * kMaximumLaguerrePolynomialDegree}
            , hat_As_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_As_"),
                      kMaximumLaguerrePolynomialDegree}
            , hat_Zs_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_Zs_"),
                      kMaximumLaguerrePolynomialDegree}
            , mug_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mug_"),
                   the_mu_extent_}
            , vparg_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "vparg_"),
                     the_vpar_extent_}
            , coeff_intdmu_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "coeff_intdmu_"),
                            the_mu_extent_}
            , coeff_intdvpar_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "coeff_intdvpar_"),
                              the_vpar_extent_}
            , rg_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "rg_"),
                  the_r_extent_}
            , nustar0_r_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar0_r_"),
                         the_r_extent_}
            , q_r_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_r_"),
                   the_r_extent_}
            , mask_buffer_r_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_buffer_r_"),
                             the_r_extent_}
            , mask_LIM_{}
            , B_norm_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "B_norm_"),
                      the_r_extent_, the_theta_extent_ + 1 /* Weird +1 */}
            , Bstar_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "Bstar_s_"),
                       the_vpar_extent_, the_r_extent_, the_theta_extent_, the_species_extent_}

            // Constant, internally computed values

            , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_j_"),
                   kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree}
            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    2 * kMaximumLaguerrePolynomialDegree - 1, kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree}
            , nustar_coeff_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar_coeff_"),
                            the_species_extent_, the_species_extent_, the_r_extent_}

            // Variable, internally computed values

            , n_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_s_"),
                   the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}
            , U_par_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_"),
                       the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}
            , T_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_s_"),
                   the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}
            , nu_ab_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu_ab_"),
                     the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_, the_species_extent_}
            , q_par_on_nT_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT_s_"),
                             the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}

            // Miscellaneous tooling

            // TODO(Etienne M): Should we partition ?
            , the_alpha_solver_{Kokkos::DefaultExecutionSpace{}} {

            // TODO(Etienne M): Say we get CPU buffers. Depending on whether
            // koliop is compiled for CPU or GPU we can either, do a shallow
            // copy, or we need to do a deep copy across memory spaces.
            // Copy::Shallow tries not to allocate more memory than necessary.
            // Copy::Deep always reallocate.
            // Both Copy::Deep and Copy::Shallow know how to handle views
            // of different layout and memory space.
            koliop::utility::Copy::Shallow(a_host_visible_comb_mat, comb_mat_);
            koliop::utility::Copy::Shallow(a_host_visible_hat_As, hat_As_);
            koliop::utility::Copy::Shallow(a_host_visible_hat_Zs, hat_Zs_);
            koliop::utility::Copy::Shallow(a_host_visible_mug, mug_);
            koliop::utility::Copy::Shallow(a_host_visible_vparg, vparg_);
            koliop::utility::Copy::Shallow(a_host_visible_coeff_intdmu, coeff_intdmu_);
            koliop::utility::Copy::Shallow(a_host_visible_coeff_intdvpar, coeff_intdvpar_);
            koliop::utility::Copy::Shallow(a_host_visible_rg, rg_);
            koliop::utility::Copy::Shallow(a_host_visible_nustar0_r, nustar0_r_);
            koliop::utility::Copy::Shallow(a_host_visible_q_r, q_r_);
            koliop::utility::Copy::Shallow(a_host_visible_mask_buffer_r, mask_buffer_r_);

            if(LIM_) {
                mask_LIM_ = koliop::utility::View::MDD2<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_LIM_"),
                                                               the_r_extent, the_theta_extent};
                koliop::utility::Copy::Shallow(a_host_visible_mask_LIM, mask_LIM_);
            }

            koliop::utility::Copy::Shallow(a_host_visible_B_norm, B_norm_);
            koliop::utility::Copy::Shallow(a_host_visible_Bstar_s, Bstar_s_);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();

            DoPrepareCj_Cjl(C_j_, C_jl_);
            DoPrepareNuStar(*this,
                            nustar_coeff_,
                            a_host_visible_hat_As,
                            a_host_visible_hat_Zs,
                            a_host_visible_rg,
                            a_host_visible_nustar0_r,
                            a_host_visible_q_r,
                            a_host_visible_mask_buffer_r);
        }

    } // namespace collision
} // namespace koliop
