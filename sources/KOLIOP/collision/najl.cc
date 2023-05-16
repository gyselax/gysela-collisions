// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/najl.h"

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

            for(OrdinalType i = 0; i < Constant::kMaximumLaguerrePolynomialDegree; ++i) {
                for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
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

            koliop::utility::Copy::Shallow(a_host_visible_C_j, C_j);

            DoPrepareCjl(a_host_visible_C_j, C_jl);
        }

        namespace {
            // NOTE: I use an unamed namespace because it is unknown to me if
            // adding static prior KOKKOS_INLINE_FUNCTION is legal.

            template <typename ValueType,
                      typename View2D,
                      typename Array1D>
            KOKKOS_INLINE_FUNCTION void
            DoComputeI(const View2D& comb_mat,
                       OrdinalType   offset_k,
                       ValueType     Xtr,
                       Array1D       I_mat) {

                static constexpr OrdinalType Xtr_pows_size = Constant::kIndexMax + 1 + 1;
                static constexpr OrdinalType J_vec_size    = Constant::kIndexMax + 2 + 1;

                ValueType Xtr_pows[Xtr_pows_size]; // std::pow(-Xtr, i);
                ValueType J_vec[J_vec_size];

                Xtr_pows[0] = 1.0;

                for(OrdinalType ind_i = 1; ind_i < Xtr_pows_size; ++ind_i) {
                    Xtr_pows[ind_i] = -Xtr * Xtr_pows[ind_i - 1];
                }

                J_vec[0] = 2.0 * (Kokkos::exp(Xtr) / Kokkos::sqrt(Xtr) - koliop::portability::MathematicalConstant<ComputeFloatType>::kSqrtPi * Kokkos::erfc(Kokkos::sqrt(Xtr)));
                J_vec[1] = koliop::portability::MathematicalConstant<ComputeFloatType>::kSqrtPi * Kokkos::erfc(Kokkos::sqrt(Xtr));

                for(OrdinalType ind_k = 2; ind_k < J_vec_size; ++ind_k) {
                    J_vec[ind_k] = (Kokkos::exp(Xtr) / Kokkos::sqrt(Xtr)) * Kokkos::abs(Xtr_pows[ind_k - 1]);
                }

                for(OrdinalType ind_k = 2; ind_k < J_vec_size; ++ind_k) {
                    J_vec[ind_k] += J_vec[ind_k - 1] * (static_cast<ValueType>(ind_k) - 1.5);
                }

                // I_mat is assumed to be zeroed.

                for(OrdinalType ind_i = 0; ind_i < Xtr_pows_size; ++ind_i) {
                    for(OrdinalType ind_k = 0; ind_k < ind_i; ++ind_k) {
                        I_mat[ind_i] += comb_mat(ind_k, ind_i) *
                                        J_vec[ind_k - offset_k + 1] *
                                        Xtr_pows[ind_i - ind_k] *
                                        Kokkos::exp(Xtr);
                    }
                }
            }

            template <typename ValueType,
                      typename View5D,
                      typename View4D,
                      typename View2D,
                      typename View1D,
                      typename Array2D>
            KOKKOS_INLINE_FUNCTION void
            DoComputeL(OrdinalType   pSpecies_id,
                       const View2D& comb_mat,
                       const View1D& hat_As,
                       const View1D& hat_Zs,
                       const View4D& n_s,
                       const View4D& T_s,
                       const View5D& nu_ab,
                       Array2D       L_mat) {
                ValueType I_mat[5 /* TODO */]{};
                ValueType Xtr = 0.0;
                DoComputeI<ValueType>(comb_mat, 0, Xtr, I_mat);
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

            const auto& the_vpar_extent = the_najl_provider.the_vpar_extent_;

            const auto& comb_mat = the_najl_provider.comb_mat_;
            const auto& hat_As   = the_najl_provider.hat_As_;
            const auto& hat_Zs   = the_najl_provider.hat_Zs_;

            const auto& pSpecies_id = the_najl_provider.pSpecies_id_;

            const auto& the_execution_space = the_najl_provider.the_execution_space_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ivpar, auto ir, auto itheta, auto iphi) {
                ValueType L_mat[Constant::kIndexMax + 1 + 1][5];

                DoComputeL<ValueType>(pSpecies_id,
                                      comb_mat,
                                      hat_As, hat_Zs,
                                      n_s, T_s,
                                      nu_ab, L_mat);
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
                   const Flo64*                                         comb_mat,
                   koliop::utility::View::MDD1<const ComputeFloatType>& hat_As,
                   koliop::utility::View::MDD1<const ComputeFloatType>& hat_Zs,
                   ExecutionSpaceType                                   an_execution_space)
            : pSpecies_id_{pSpecies_id}
            , the_vpar_extent_{the_vpar_extent}

            , comb_mat_{koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL2<const Flo64>,
                                                             koliop::utility::View::MDD2>(comb_mat, "comb_mat_", 2 * Constant::kMaximumLaguerrePolynomialDegree, 2 * Constant::kMaximumLaguerrePolynomialDegree)}
            , hat_As_{hat_As}
            , hat_Zs_{hat_Zs}
            , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_j_"),
                   static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree),
                   static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree)}
            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    static_cast<std::size_t>(2 * Constant::kMaximumLaguerrePolynomialDegree - 1),
                    static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree),
                    static_cast<std::size_t>(Constant::kMaximumLaguerrePolynomialDegree)}

            , the_execution_space_{an_execution_space} {

            // TODO(Etienne M): Observe how the C_j_ and C_jl_ values are
            // constant. they could be compile time computed or hardcoded.
            DoPrepareCj_Cjl(C_j_, C_jl_);
        }

        void Najl::Compute(OrdinalType                                          the_block_first_r,
                           OrdinalType                                          the_block_first_theta,
                           OrdinalType                                          the_block_first_phi,
                           const koliop::utility::View::MDD4<ComputeFloatType>& n_s,
                           const koliop::utility::View::MDD4<ComputeFloatType>& T_s,
                           const koliop::utility::View::MDD5<ComputeFloatType>& nu_ab,
                           Constant::ScalarBlockTridiagonalByBlockMatrixType&   the_Najl) const {

            KOLIOP_ASSERT(the_Najl.size() != 0);

            // TODO(Etienne M): Other tests

            // TODO(Etienne M): It is not currently possible to do something
            // like:
            // Kokkos::exp(T{}) where T is a KokkosBatched::Vector !

            // VectorBlockTridiagonalByBlockMatrixType the_Najl_view{reinterpret_cast<typename Constant::BatchTraitsType::ThreadVectorType*>(the_Najl.data()),
            //                                                       static_cast<std::size_t>(Constant::kBatchCount),
            //                                                       static_cast<std::size_t>(the_vpar_extent_)};

            ScalarBlockTridiagonalByBlockMatrixType the_Najl_view{the_Najl.data(),
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
