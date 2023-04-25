// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

#include "KOLIOP/portability/cmath.h"
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
                for(OrdinalType i = 0; i <= (j_tmp + l_tmp); ++i) {
                    // This seems unnecessary ?
                    a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                }

                for(OrdinalType alpha = 0; alpha <= j_tmp; ++alpha) {
                    for(OrdinalType beta = 0; beta <= l_tmp; ++beta) {
                        a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) += a_host_visible_C_j(alpha, j_tmp) *
                                                                           a_host_visible_C_j(beta, l_tmp);
                    }
                }
            };

            for(OrdinalType i = 0; i < Operator::kMaximumLaguerrePolynomialDegree; ++i) {
                for(OrdinalType j = 0; j < Operator::kMaximumLaguerrePolynomialDegree; ++j) {
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

            for(OrdinalType j = 0; j < Operator::kMaximumLaguerrePolynomialDegree; ++j) {
                a_host_visible_C_j(0, j) = 1.0;
                a_host_visible_C_j(1, j) = static_cast<ComputeFloatType>(-static_cast<int64_t>(j));
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

        template <typename View3D,
                  typename View1D>
        static inline void
        DoPrepareNuStar(const Operator& an_operator,
                        View3D&&        nustar_coeff,
                        const View1D&   hat_As,
                        const View1D&   hat_Zs,
                        const View1D&   rg,
                        const View1D&   nustar0_r,
                        const View1D&   q_r,
                        const View1D&   mask_buffer_r) {

            // TODO(Etienne M): I observe some slight numerical discrepancy
            // (1e-14 ~ ) when comparing the result of Gysela and koliop.

            const auto a_host_visible_hat_As        = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_As);
            const auto a_host_visible_hat_Zs        = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_Zs);
            const auto a_host_visible_rg            = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), rg);
            const auto a_host_visible_nustar0_r     = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), nustar0_r);
            const auto a_host_visible_q_r           = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), q_r);
            const auto a_host_visible_mask_buffer_r = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), mask_buffer_r);
            const auto a_host_visible_nustar_coeff  = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), nustar_coeff);

            koliop::utility::View::MHL1<ComputeFloatType> a_host_visible_mask_space{Kokkos::view_alloc(Kokkos::WithoutInitializing, "a_host_visible_mask_space"),
                                                                                    static_cast<std::size_t>(an_operator.the_r_extent_)};

            if(a_host_visible_q_r(0) < koliop::portability::GyselaConstant<ComputeFloatType>::kInfinity) {
                for(OrdinalType i = 0; i < an_operator.the_r_extent_; ++i) {
                    a_host_visible_mask_space(i) = static_cast<ComputeFloatType>(1.0) / a_host_visible_q_r(i);
                }
            }

            if(!(an_operator.LIM_ || an_operator.SOL_)) {
                for(OrdinalType i = 0; i < an_operator.the_r_extent_; ++i) {
                    a_host_visible_mask_space(i) *= (static_cast<ComputeFloatType>(1.0) - a_host_visible_mask_buffer_r(i));
                }
            }

            const auto R0_inverse = static_cast<ComputeFloatType>(1.0) / an_operator.R0_;

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
            for(OrdinalType ispeciesB = 0; ispeciesB < an_operator.the_species_extent_; ++ispeciesB) {
                for(OrdinalType ispeciesA = 0; ispeciesA < an_operator.the_species_extent_; ++ispeciesA) {
                    const ComputeFloatType hat_Aa = a_host_visible_hat_As(ispeciesA);
                    const ComputeFloatType hat_Za = a_host_visible_hat_Zs(ispeciesA);
                    const ComputeFloatType hat_Ab = a_host_visible_hat_As(ispeciesB);
                    const ComputeFloatType hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                    const ComputeFloatType AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa + static_cast<ComputeFloatType>(1.0) / hat_Ab);

                    for(OrdinalType ir = 0; ir < an_operator.the_r_extent_; ++ir) {
                        // Computation of
                        // AD(r) = sqrt(2) * eps ^ (3 / 2) / (q R0_)*nustaR0_(r)
                        // which is independent on the species considered (eps = r/R0_)
                        const ComputeFloatType eps = a_host_visible_rg(ir) * R0_inverse;

                        const ComputeFloatType AD = a_host_visible_nustar0_r(ir) *
                                                    a_host_visible_mask_space(ir) *
                                                    // TODO(Etienne M): SIMPLIFY
                                                    // sqrt(2)*(rg*R0_inverse)*sqrt(rg*R0_inverse)*R0_inverse_ =
                                                    // sqrt(2)*pow(r, 5/2)*pow(rg, 3/2)
                                                    // sqrt(2)*pow(r, 5/2) is constant here.
                                                    koliop::portability::MathematicalConstant<ComputeFloatType>::kSQRT2 *
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

        Operator::Operator(uint64_t     pSpecies_id,
                           uint64_t     mu_id,
                           uint64_t     the_mu_extent,
                           uint64_t     the_vpar_extent,
                           uint64_t     the_r_extent,
                           uint64_t     the_theta_extent,
                           uint64_t     the_phi_extent,
                           uint64_t     the_species_extent,
                           Flo64        R0,
                           bool         collision_interspecies,
                           bool         SOL,
                           bool         LIM,
                           uint64_t     ir_SOL_separatrix,
                           Flo64        dt,
                           const Flo64* comb_mat,
                           const Flo64* hat_As,
                           const Flo64* hat_Zs,
                           const Flo64* mug,
                           const Flo64* vparg,
                           const Flo64* coeff_intdmu,
                           const Flo64* coeff_intdvpar,
                           const Flo64* rg,
                           const Flo64* nustar0_r,
                           const Flo64* q_r,
                           const Flo64* mask_buffer_r,
                           const Flo64* mask_LIM,
                           const Flo64* B_norm,
                           const Flo64* Bstar_s)
            // Constant, externally computed values
            : pSpecies_id_{static_cast<OrdinalType>(pSpecies_id)}
            , mu_id_{static_cast<OrdinalType>(mu_id)}
            , the_mu_extent_{static_cast<OrdinalType>(the_mu_extent)}
            , the_vpar_extent_{static_cast<OrdinalType>(the_vpar_extent)}
            , the_r_extent_{static_cast<OrdinalType>(the_r_extent)}
            , the_theta_extent_{static_cast<OrdinalType>(the_theta_extent)}
            , the_phi_extent_{static_cast<OrdinalType>(the_phi_extent)}
            , the_species_extent_{static_cast<OrdinalType>(the_species_extent)}
            , R0_{static_cast<ComputeFloatType>(R0)}
            , collision_interspecies_{collision_interspecies}
            , SOL_{SOL}
            , LIM_{LIM}
            , ir_SOL_separatrix_{static_cast<OrdinalType>(ir_SOL_separatrix)}
            , dt_{static_cast<ComputeFloatType>(dt)}
            , comb_mat_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL2<const Flo64>, koliop::utility::View::MDD2>(comb_mat, "comb_mat_", 2 * kMaximumLaguerrePolynomialDegree, 2 * kMaximumLaguerrePolynomialDegree)}
            , hat_As_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(hat_As, "hat_As_", kMaximumLaguerrePolynomialDegree)}
            , hat_Zs_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(hat_Zs, "hat_Zs_", kMaximumLaguerrePolynomialDegree)}
            , mug_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(mug, "mug_", the_mu_extent_)}
            , vparg_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(vparg, "vparg_", the_vpar_extent_)}
            , coeff_intdmu_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(coeff_intdmu, "coeff_intdmu_", the_mu_extent_)}
            , coeff_intdvpar_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(coeff_intdvpar, "coeff_intdvpar_", the_vpar_extent_)}
            , rg_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(rg, "rg_", the_r_extent_)}
            , nustar0_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(nustar0_r, "nustar0_r_", the_r_extent_)}
            , q_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(q_r, "q_r_", the_r_extent_)}
            , mask_buffer_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL1<const Flo64>, koliop::utility::View::MDD1>(mask_buffer_r, "mask_buffer_r_", the_r_extent_)}
            , mask_LIM_{LIM_ ? koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL2<const Flo64>, koliop::utility::View::MDD2>(mask_LIM, "mask_LIM_", the_r_extent, the_theta_extent) :
                               koliop::utility::View::MDD2<Flo64>{}}
            , B_norm_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL2<const Flo64>, koliop::utility::View::MDD2>(B_norm, "B_norm_", the_r_extent_, the_theta_extent_ + 1 /* Weird +1 */)}
            , Bstar_s_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL4<const Flo64>, koliop::utility::View::MDD4>(Bstar_s, "Bstar_s_", the_vpar_extent_, the_r_extent_, the_theta_extent_, the_species_extent_)}

            // Constant, internally computed values

            , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_j_"),
                   static_cast<std::size_t>(kMaximumLaguerrePolynomialDegree), static_cast<std::size_t>(kMaximumLaguerrePolynomialDegree)}
            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    static_cast<std::size_t>(2 * kMaximumLaguerrePolynomialDegree - 1), static_cast<std::size_t>(kMaximumLaguerrePolynomialDegree), static_cast<std::size_t>(kMaximumLaguerrePolynomialDegree)}
            , nustar_coeff_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar_coeff_"),
                            static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_r_extent_)}

            // Variable, internally computed values

            , n_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_s_"),
                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , U_par_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_"),
                       static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , T_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_s_"),
                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , nu_ab_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu_ab_"),
                     static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_species_extent_)}
            , q_par_on_nT_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT_s_"),
                             static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}

            // Miscellaneous tooling

            // TODO(Etienne M): Should we partition ?
            , the_alpha_solver_{Kokkos::DefaultExecutionSpace{}} {

            // TODO(Etienne M): Should we fence after the copies ? We *may* need
            // that if we pass an execution space to the deep copy.
            // Kokkos::fence();

            DoPrepareCj_Cjl(C_j_, C_jl_);
            DoPrepareNuStar(*this,
                            nustar_coeff_,
                            hat_As_,
                            hat_Zs_,
                            rg_,
                            nustar0_r_,
                            q_r_,
                            mask_buffer_r_);
        }

    } // namespace collision
} // namespace koliop
