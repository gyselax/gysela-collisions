// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/najl.h"

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View3D>
        static inline void
        DoPrepareCjl(View3D& C_jl) {

            // TODO(Etienne M): This small matrix is know at compile time !

            // a_host_visible_C_j is zero initialized.
            koliop::utility::View::MHL2<ComputeFloatType> a_host_visible_C_j{Kokkos::view_alloc("C_j"),
                                                                             static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree),
                                                                             static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree)};

            for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                a_host_visible_C_j(0, j) = 1.0;
                a_host_visible_C_j(1, j) = static_cast<ComputeFloatType>(-static_cast<int64_t>(j));
            }

            if(Constant::kMaximumLaguerrePolynomialDegree > 2) {
                a_host_visible_C_j(2, 2) = 0.5;
                if(Constant::kMaximumLaguerrePolynomialDegree > 3) {
                    // NOTE: Assuming Constant::kMaximumLaguerrePolynomialDegree = 3, this
                    // is never evaluated.
                    a_host_visible_C_j(2, 3) = 1.5;
                    a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                }
            }

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

            for(OrdinalType i = 0; i < Constant::kMaximumLaguerrePolynomialDegree; ++i) {
                for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                    CoreLoop(i, j);
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_jl, C_jl);
        }

        namespace {
            // NOTE: I use an unamed namespace because it is unknown to me if
            // adding static prior KOKKOS_INLINE_FUNCTION is legal.

            template <typename ValueType,
                      typename View2D,
                      typename Array1D>
            KOKKOS_INLINE_FUNCTION void
            DoComputeI(const View2D& comb_mat,
                       OffsetType    offset,
                       ValueType     Xtr,
                       ValueType     sqrtXtr,
                       ValueType     sqrtXtr_inv,
                       Array1D       I_mat) {

                static constexpr OrdinalType the_Xtr_pows_size = Constant::kIndexMax + 1 + 1;
                static constexpr OrdinalType the_J_vec_size    = Constant::kIndexMax + 2 + 1;

                ValueType Xtr_pows[the_Xtr_pows_size]; // std::pow(-Xtr, i);
                ValueType J_vec[the_J_vec_size];

                Xtr_pows[0] = 1.0;

                for(OrdinalType ind_i = 1; ind_i < the_Xtr_pows_size; ++ind_i) {
                    Xtr_pows[ind_i] = -Xtr * Xtr_pows[ind_i - 1];
                }

                const ValueType the_exp_Xtr                = Kokkos::exp(Xtr);
                const ValueType the_sqrtXtr_inv_on_exp_Xtr = sqrtXtr_inv / the_exp_Xtr;

                J_vec[1] = koliop::portability::MathematicalConstant<ValueType>::kSqrtPi * Kokkos::erfc(sqrtXtr);
                J_vec[0] = 2.0 * (the_sqrtXtr_inv_on_exp_Xtr - J_vec[1]);

                for(OrdinalType ind_k = 2; ind_k < the_J_vec_size; ++ind_k) {
                    J_vec[ind_k] = the_sqrtXtr_inv_on_exp_Xtr * Kokkos::abs(Xtr_pows[ind_k - 1]);
                }

                for(OrdinalType ind_k = 2; ind_k < the_J_vec_size; ++ind_k) {
                    J_vec[ind_k] += J_vec[ind_k - 1] * (static_cast<ValueType>(ind_k) - 1.5);
                }

                // Assumes I_mat is zeroed.
                // I_mat(:) = 0.0;

                for(OrdinalType ind_i = 0; ind_i < the_Xtr_pows_size; ++ind_i) {
                    for(OrdinalType ind_k = 0; ind_k <= ind_i; ++ind_k) {
                        // NOTE: The numerical stability of this whole operation
                        // is not good !
                        // TODO(Etienne M): Analyze the numerical stability of
                        // the Najl
                        I_mat[ind_i] += (comb_mat(ind_k, ind_i) *
                                         Xtr_pows[ind_i - ind_k]) *
                                        (J_vec[ind_k + offset] *
                                         the_exp_Xtr);
                    }
                }
            }

            template <typename ValueType,
                      typename View2D,
                      typename View1D0,
                      typename View1D1,
                      typename Array2D>
            KOKKOS_INLINE_FUNCTION void
            DoComputeL(OrdinalType    the_species_extent,
                       OrdinalType    pSpecies_id,
                       bool           collision_interspecies,
                       const View2D&  comb_mat,
                       const View1D0& hat_As,
                       const View1D0& hat_Zs,
                       const View1D1& local_n_s,
                       const View1D1& local_T_s,
                       ValueType      nu,
                       ValueType      vpar2_on_Ta,
                       Array2D        L_mat) {

                const ValueType hat_Aa = hat_As(pSpecies_id);
                const ValueType hat_Za = hat_Zs(pSpecies_id);

                // Assumes L_mat is zeroed.
                // L_mat(:,:) = 0.0;

                for(OrdinalType ispeciesB = 0; ispeciesB < the_species_extent; ++ispeciesB) {
                    if(!(collision_interspecies || pSpecies_id == ispeciesB)) {
                        continue;
                    }

                    const ValueType hat_Ab = hat_As(ispeciesB);
                    const ValueType hat_Zb = hat_Zs(ispeciesB);

                    const ValueType coeff_L = (0.75 * koliop::portability::MathematicalConstant<ValueType>::kSqrtPi * koliop::portability::MathematicalConstant<ValueType>::kSqrt2) *
                                              nu * (hat_Zb * hat_Zb * local_n_s(ispeciesB)) / (hat_Za * hat_Za * local_n_s(pSpecies_id));

                    const ValueType vTb_on_vta_2 = (hat_Aa * local_T_s(ispeciesB)) / (hat_Ab * local_T_s(pSpecies_id));

                    {
                        const ValueType Xtr         = vpar2_on_Ta * 0.5 + (9.0 * koliop::portability::MathematicalConstant<ValueType>::kPi / 16.0) * vTb_on_vta_2;
                        const ValueType sqrtXtr     = Kokkos::sqrt(Xtr);
                        const ValueType sqrtXtr_inv = 1.0 / sqrtXtr;

                        ValueType I_mat[Constant::kIndexMax + 1 + 1]{};

                        DoComputeI(comb_mat, 1, Xtr, sqrtXtr, sqrtXtr_inv, I_mat);

                        for(OrdinalType ind_i = 0; ind_i <= Constant::kIndexMax; ++ind_i) {
                            L_mat[0][ind_i] += coeff_L * I_mat[ind_i];
                        }

                        L_mat[1][0] -= coeff_L * sqrtXtr_inv;
                    }
                    {
                        const ValueType Xtr         = vpar2_on_Ta * 0.5 + 2.1 * vTb_on_vta_2;
                        const ValueType sqrtXtr     = Kokkos::sqrt(Xtr);
                        const ValueType sqrtXtr_inv = 1.0 / sqrtXtr;

                        ValueType I_mat[Constant::kIndexMax + 1 + 1]{};
                        DoComputeI(comb_mat, 0, Xtr, sqrtXtr, sqrtXtr_inv, I_mat);

                        for(OrdinalType ind_i = 0; ind_i <= Constant::kIndexMax; ++ind_i) {
                            L_mat[2][ind_i] -= coeff_L * I_mat[ind_i];
                        }

                        L_mat[3][0] += (sqrtXtr_inv * sqrtXtr_inv * sqrtXtr_inv) * (coeff_L * vpar2_on_Ta * 0.5);
                        L_mat[4][0] += (sqrtXtr_inv * sqrtXtr_inv * sqrtXtr_inv) * (coeff_L);
                    }
                }

                L_mat[1][0] += L_mat[0][0];

                for(OrdinalType ind_i = 1; ind_i <= Constant::kIndexMax; ++ind_i) {
                    L_mat[1][ind_i] += L_mat[0][ind_i] -
                                       L_mat[0][ind_i - 1] * static_cast<ValueType>(ind_i);
                }

                L_mat[3][0] += L_mat[2][0] * vpar2_on_Ta * 0.5 +
                               L_mat[2][1] -
                               L_mat[2][0];

                for(OrdinalType ind_i = 1; ind_i <= Constant::kIndexMax; ++ind_i) {
                    L_mat[3][ind_i] += -L_mat[2][ind_i - 1] * static_cast<ValueType>(ind_i) * vpar2_on_Ta * 0.5 +
                                       L_mat[2][ind_i] * (vpar2_on_Ta * 0.5 - static_cast<ValueType>(ind_i + 1)) +
                                       L_mat[2][ind_i + 1];
                }

                L_mat[4][0] += L_mat[2][0];

                for(OrdinalType ind_i = 1; ind_i <= Constant::kIndexMax; ++ind_i) {
                    L_mat[4][ind_i] += L_mat[2][ind_i] -
                                       L_mat[2][ind_i - 1] * static_cast<ValueType>(ind_i);
                }
            }

        } // namespace

        template <typename View6D,
                  typename View5D,
                  typename View4D>
        static inline void
        DoComputeNajl(const Najl&   the_najl_provider,
                      OrdinalType   the_block_first_r,
                      OrdinalType   the_block_first_theta,
                      OrdinalType   the_block_first_phi,
                      const View4D& n_s,
                      const View4D& T_s,
                      const View5D& nu_ab,
                      View6D&&      the_Najl) {

            // NOTE: We use ValueType due to an hypothetical future Kokkos
            // release that would enable Kokkos::exp() on KokkosBatched::Vector.

            using ValueType = typename std::remove_reference_t<View6D>::value_type;

            const auto& the_vpar_extent    = the_najl_provider.the_vpar_extent_;
            const auto& the_species_extent = the_najl_provider.the_species_extent_;

            const auto& comb_mat = the_najl_provider.comb_mat_;
            const auto& hat_As   = the_najl_provider.hat_As_;
            const auto& hat_Zs   = the_najl_provider.hat_Zs_;
            const auto& vparg    = the_najl_provider.vparg_;

            const auto& C_jl = the_najl_provider.C_jl_;

            const auto& pSpecies_id            = the_najl_provider.pSpecies_id_;
            const auto& collision_interspecies = the_najl_provider.collision_interspecies_;

            const auto& the_execution_space = the_najl_provider.the_execution_space_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ivpar, auto ir, auto itheta, auto iphi) {

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const OrdinalType the_block_local_index_r     = ir - the_block_first_r;
                const OrdinalType the_block_local_index_theta = itheta - the_block_first_theta;
                const OrdinalType the_block_local_index_phi   = iphi - the_block_first_phi;

                const OffsetType the_flattened_index = Constant::kBlockExtentR * Constant::kBlockExtentTheta * the_block_local_index_phi +
                                                       Constant::kBlockExtentR * the_block_local_index_theta +
                                                       the_block_local_index_r;

                const OrdinalType the_batch_index         = the_flattened_index / Constant::BatchTraitsType::kBatchSize;
                const OrdinalType the_batch_element_index = the_flattened_index % Constant::BatchTraitsType::kBatchSize;

                // NOTE: The memset(0) of the local_Nn is done in Najl::Compute.
                auto local_N0 = Kokkos::subview(the_Najl,
                                                the_batch_index,
                                                ivpar,
                                                0,
                                                Kokkos::ALL(), // j
                                                Kokkos::ALL(), // l
                                                the_batch_element_index);

                auto local_N1 = Kokkos::subview(the_Najl,
                                                the_batch_index,
                                                ivpar,
                                                1,
                                                Kokkos::ALL(), // j
                                                Kokkos::ALL(), // l
                                                the_batch_element_index);

                auto local_N2 = Kokkos::subview(the_Najl,
                                                the_batch_index,
                                                ivpar,
                                                2,
                                                Kokkos::ALL(), // j
                                                Kokkos::ALL(), // l
                                                the_batch_element_index);

                const auto local_n_s = Kokkos::subview(n_s,
                                                       ir,
                                                       itheta,
                                                       iphi,
                                                       Kokkos::ALL());

                const auto local_T_s = Kokkos::subview(T_s,
                                                       ir,
                                                       itheta,
                                                       iphi,
                                                       Kokkos::ALL());

                const ValueType vpar2_on_Ta = vparg(ivpar) * vparg(ivpar) / local_T_s(pSpecies_id);

                // NOTE: Careful, L_mat is layout right
                ValueType L_mat[5][Constant::kIndexMax + 1 + 1]{};

                DoComputeL<ValueType>(the_species_extent,
                                      pSpecies_id,
                                      collision_interspecies,
                                      comb_mat,
                                      hat_As, hat_Zs,
                                      local_n_s, local_T_s,
                                      nu_ab(ir, itheta, iphi, pSpecies_id, pSpecies_id),
                                      vpar2_on_Ta,
                                      L_mat);

                for(OrdinalType ind_i = 0; ind_i <= Constant::kIndexMax; ++ind_i) {
                    ValueType N_coeff = L_mat[1][ind_i] * 2.0 +
                                        L_mat[2][ind_i] * (3.0 - vpar2_on_Ta) +
                                        L_mat[4][ind_i] * (2.0 + vpar2_on_Ta);

                    if(ind_i != Constant::kIndexMax) {
                        for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                            for(OrdinalType j = 1; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                local_N0(j, l) -= static_cast<ValueType>(j) * N_coeff * C_jl(ind_i, j - 1, l);
                            }
                        }
                    }

                    N_coeff -= (L_mat[0][ind_i] + L_mat[2][ind_i + 1]) * 2.0;

                    for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                        for(OrdinalType j = 1; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                            local_N0(j, l) += static_cast<ValueType>(j) * N_coeff * C_jl(ind_i, j, l);
                        }
                    }

                    N_coeff = -L_mat[0][ind_i] +
                              L_mat[1][ind_i] -
                              L_mat[2][ind_i + 1] +
                              L_mat[2][ind_i] * (2.0 - vpar2_on_Ta * 0.5) +
                              L_mat[3][ind_i];

                    if(ind_i != Constant::kIndexMax) {
                        for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                            local_N1(0, l) += vparg(ivpar) * C_jl(ind_i, 0, l) * N_coeff;
                            for(OrdinalType j = 1; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                local_N1(j, l) += vparg(ivpar) * (C_jl(ind_i, j, l) * N_coeff +
                                                                  2.0 * static_cast<ValueType>(j) * L_mat[2][ind_i] * (C_jl(ind_i, j, l) - C_jl(ind_i, j - 1, l)));
                            }
                        }

                    } else {
                        for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                            for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                local_N1(j, l) += vparg(ivpar) *
                                                  C_jl(Constant::kIndexMax, j, l) *
                                                  (N_coeff +
                                                   2.0 * static_cast<ValueType>(j) * L_mat[2][Constant::kIndexMax]);
                            }
                        }
                    }

                    N_coeff = local_T_s(pSpecies_id) * (L_mat[0][ind_i] +
                                                        L_mat[2][ind_i] * vpar2_on_Ta * 0.5);

                    for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                        for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                            local_N2(j, l) += C_jl(ind_i, j, l) * N_coeff;
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{the_execution_space,
                                                                                             {0,
                                                                                              the_block_first_r,
                                                                                              the_block_first_theta,
                                                                                              the_block_first_phi},
                                                                                             {the_vpar_extent,
                                                                                              the_block_first_r + Constant::kBlockExtentR,
                                                                                              the_block_first_theta + Constant::kBlockExtentTheta,
                                                                                              the_block_first_phi + Constant::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeNajl",
                                 a_policy, the_kernel);
        }

        Najl::Najl(OrdinalType                                          pSpecies_id,
                   OrdinalType                                          the_vpar_extent,
                   OrdinalType                                          the_species_extent,
                   bool                                                 collision_interspecies,
                   const Flo64*                                         comb_mat,
                   koliop::utility::View::MDD1<const ComputeFloatType>& hat_As,
                   koliop::utility::View::MDD1<const ComputeFloatType>& hat_Zs,
                   koliop::utility::View::MDD1<const ComputeFloatType>& vparg,
                   ExecutionSpaceType                                   an_execution_space)
            : pSpecies_id_{pSpecies_id}
            , the_vpar_extent_{the_vpar_extent}
            , the_species_extent_{the_species_extent}
            , collision_interspecies_{collision_interspecies}

            , comb_mat_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL2<const Flo64>,
                                                             koliop::utility::View::MDD2>(comb_mat, "comb_mat_", 2 * Constant::kMaximumLaguerrePolynomialDegree, 2 * Constant::kMaximumLaguerrePolynomialDegree)}
            , hat_As_{hat_As}
            , hat_Zs_{hat_Zs}
            , vparg_{vparg}

            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    static_cast<std::size_t>(2 * Constant::kMaximumLaguerrePolynomialDegree - 1),
                    static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree),
                    static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree)}

            , the_execution_space_{an_execution_space} {

            // TODO(Etienne M): Observe how the C_j_ and C_jl_ values are
            // constant. they could be compile time computed or hardcoded.
            DoPrepareCjl(C_jl_);
        }

        void Najl::Compute(OrdinalType                                          the_block_first_r,
                           OrdinalType                                          the_block_first_theta,
                           OrdinalType                                          the_block_first_phi,
                           const koliop::utility::View::MDD4<ComputeFloatType>& n_s,
                           const koliop::utility::View::MDD4<ComputeFloatType>& T_s,
                           const koliop::utility::View::MDD5<ComputeFloatType>& nu_ab,
                           typename MatrixKindType::ScalarType&                 the_Najl) const {

            KOLIOP_ASSERT(the_Najl.size() != 0);

            // TODO(Etienne M): Other tests

            // TODO(Etienne M): what is the perf of such memset:
            Kokkos::deep_copy(/* TODO(Etienne M): execution space ,*/ the_Najl, 0.0);

            // TODO(Etienne M): It is not currently possible to do something
            // like:
            // Kokkos::exp(T{}) where T is a KokkosBatched::Vector !

            // typename MatrixKindType::VectorType the_Najl_view{reinterpret_cast<typename Constant::BatchTraitsType::ThreadVectorType*>(the_Najl.data()),
            //                                                   static_cast<std::size_t>(Constant::kBatchCount),
            //                                                   static_cast<std::size_t>(the_vpar_extent_)};

            typename MatrixKindType::ScalarType the_Najl_view{the_Najl.data(),
                                                              static_cast<std::size_t>(Constant::kBatchCount),
                                                              static_cast<std::size_t>(the_vpar_extent_)};

            DoComputeNajl(*this,
                          the_block_first_r,
                          the_block_first_theta,
                          the_block_first_phi,
                          n_s,
                          T_s,
                          nu_ab,
                          the_Najl_view);
        }

    } // namespace collision
} // namespace koliop
