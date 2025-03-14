// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_NAJL_H
#define KOLIOP_COLLISION_NAJL_H

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {
        namespace detail {

            /// Computation of the coefficient C(i,j,l) corresponding to
            /// PjPl = sum_i C(i,j,l) u^i
            /// C(i,j,l) = sum_alpha sum_beta Cj(alpha) Cl(beta) delta(alpha +beta -i)
            /// where Cj and Cl correspond respectively to the coefficient associated
            /// to Pj and Pl.
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

                // a_host_visible_C_jl is zero initialized if C_jl is not on the
                // host and will not be if C_jl is on the host (that would mean
                // overwriting the content of C_jl). This asymmetry is somewhat
                // dirty and requires a branch if one whishes not to do
                // unnecessary zeroing. For instance, the code below is great on
                // when the device is the CPU, but leads to a redundant memset
                // when the device is the GPU.
                auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl);
                if constexpr(Kokkos::is_always_assignable_v<std::remove_reference_t<decltype(C_jl)>,
                                                            std::remove_reference_t<decltype(a_host_visible_C_jl)>>) {
                    Kokkos::deep_copy(a_host_visible_C_jl, static_cast<ComputeFloatType>(0.0));
                }

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

                const Constant::DeviceExecutionSpaceType an_execution_space{};

                koliop::utility::Copy::Shallow(an_execution_space,
                                               a_host_visible_C_jl,
                                               C_jl);

                an_execution_space.fence();
            }

            namespace {
                // NOTE: I use an unamed namespace because it is unknown to me if
                // adding static prior KOKKOS_INLINE_FUNCTION is legal.

                /// Computation of I(n, i, Xtr) = int from 0 to infinity of
                ///   dU U^i exp(-U) / (U+Xtr)^(n+0.5)
                ///  using the recursive computation of
                ///   J(k, X_tr) = \int_sqrt(X_tr)^\infty 2 v^(2k) exp(-v**2) dv
                /// where k is between -1 and index_max+1 (in the code, the
                /// index k in J_vec is shifted to start from 0).
                /// When Xtr > 5 (n+1/2) (index_max_+2), the 3rd order
                /// asymptotic approximation
                ///   I(n, i, Xtr) = i! / Xtr**(n+1/2) *
                ///     ( 1 - (n+1/2)(i+1)/Xtr * (1 - (1/2)(n+3/2)(i+2)/Xtr ) )
                /// is used instead.
                /// Since Xtr might contain both values above and below this
                /// threshold, the code separates the values belonging to either
                /// category, computes I for the 2 groups and finally build a
                /// common result array (if necessary).
                ///
                template <typename ValueType,
                          typename Array1D>
                KOKKOS_INLINE_FUNCTION void
                DoComputeI(OffsetType offset_k,
                           ValueType  Xtr,
                           ValueType  sqrtXtr,
                           ValueType  sqrtXtr_inv,
                           Array1D    I_mat) {

                    static constexpr OrdinalType Xtr_pows_size = Constant::kIndexMax + 1 + 1;

                    if(Xtr > ((offset_k + 0.5) * (5 * Xtr_pows_size))) {
                        // Xtr is less than (15 or 45).
                        // NOTE: This nasty branch is a followup on the work
                        // done by Peter and Kevin regarding a bug when Xtr
                        // becomes too large.
                        // See Gysela fortran commit 175f2f4c
                        // This approximation is a lot less costly than the
                        // above method, and often, we can use it, reducing the
                        // compute load.
                        //
                        // We need: int(U^d exp(-U)/(U+Xtr)^(offset_k+0.5), dU, 0 to infty)
                        // With approx valid for
                        //      Xtr > 200
                        // we have:
                        //      I(offset_k, i, Xtr) = i! * Kokkos::pow(Xtr, -offset_k - 0.5)
                        //
                        // With a 3rd order approx valid for
                        //      Xtr > ((offset_k + 0.5) * (5 * (kIndexMax + 2)))
                        // we have:
                        //      I(offset_k, i, Xtr) = i! * Kokkos::pow(Xtr, -offset_k - 0.5) *
                        //                     ( 1 -       (offset_k+0.5)*(i+1)/Xtr *
                        //                      (1 - (0.5)*(offset_k+1.5)*(i+2)/Xtr ) );
                        //
                        static constexpr ValueType kFactorial[Xtr_pows_size] = {1, 1, 2, 6, 24, 120};

                        const ValueType offset_k_float = static_cast<ValueType>(offset_k);
#if defined(KOLIOP_COMPILER_IS_GCC)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif
#pragma unroll
                        for(OrdinalType ind_i = 0; ind_i < Xtr_pows_size; ++ind_i) {
                            // clang-format off
                            I_mat[ind_i] = (kFactorial[ind_i] * Kokkos::pow(Xtr, -offset_k_float - 0.5)) *
                                           (1.0 - ((offset_k_float + 0.5) * (static_cast<ValueType>(ind_i) + 1)) / Xtr *
                                           (1.0 - ((offset_k_float + 1.5) * (static_cast<ValueType>(ind_i) + 2)) / Xtr * 0.5));
                            // clang-format on
                        }
#if defined(KOLIOP_COMPILER_IS_GCC)
    #pragma GCC diagnostic pop
#endif
                    } else {
                        // NOTE: Numerical accuracy of this whole branch's computation
                        // isn't good!
                        // The error comes from the fact that the contraction of
                        // floating point operations into FMA changes the
                        // accuracy of the computation. Coupled with the different
                        // behavior of the exp and erfc function (GPU vs GLibC), we
                        // get a very significant error (up to ~1E-5 relative).
                        // Initially the error is not large (~1E-14 relative,
                        // measured out of erfc) but the whole Najl coefficient
                        // computation is extraordinarily powerful when it copy to
                        // spreading and amplifying errors.
                        // Also, by setting Xtr, sqrtXtr, sqrtXtr_inv to constant
                        // and similar value say 2.0,3.0,4.0 on the Fortran and C++
                        // version of the operator, we observe below 10^-10 abs and
                        // relative errors which is good but weird.
                        // Finally, note the comment below regarding the changes to
                        // the computation of I_mat and the error that we then get.
                        // To mitigate that divergence from the reference CrayFTN
                        // Gysela, two facts, use:
                        // - #pragma clang fp contract(on) : instead of the fast or
                        // fast-honor-pragmas which is the default for GPU on Clang.
                        // - Use more accurate transcendental function
                        // implementation (relative to GLibC).
                        // NOTE: theses error would also obviously manifest
                        // themselve if we changed the compiler from gfortran to
                        // ifort or crayftn and as such, trying to be bit-perfect
                        // with the CrayFTN reference is but an illusion (and incurs
                        // a performance penalty).

                        static constexpr OrdinalType J_vec_size = Constant::kIndexMax + 2 + 1;

                        // Precompute kCombMat(n, k): n! /(k! * (n-k)!)
                        // AKA: binomial coeff/pascal's triangle
                        // n are the ind_i
                        // k are the ind_k
                        // clang-format off
                        static constexpr ValueType kCombMat[Xtr_pows_size][Xtr_pows_size] =
                            {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                             {1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                             {1.0, 2.0, 1.0, 0.0, 0.0, 0.0},
                             {1.0, 3.0, 3.0, 1.0, 0.0, 0.0},
                             {1.0, 4.0, 6.0, 4.0, 1.0, 0.0},
                             {1.0, 5.0, 10.0, 10.0, 5.0, 1.0}};
                        // clang-format on

                        ValueType Xtr_pows[Xtr_pows_size]; // std::pow(-Xtr, i);
                        ValueType J_vec[J_vec_size];

                        // for(OrdinalType ind_i = 0; ind_i < Xtr_pows_size; ++ind_i) {
                        //     Xtr_pows[ind_i] = Kokkos::pow(-Xtr, static_cast<ValueType>(ind_i));
                        // }

                        Xtr_pows[0] = 1.0;

                        for(OrdinalType ind_i = 1; ind_i < Xtr_pows_size; ++ind_i) {
                            Xtr_pows[ind_i] = -Xtr * Xtr_pows[ind_i - 1];
                        }

                        const ValueType exp_Xtr                = Kokkos::exp(Xtr);
                        const ValueType sqrtXtr_inv_on_exp_Xtr = sqrtXtr_inv / exp_Xtr;
                        const ValueType erfc_sqrt_Xtr          = Kokkos::erfc(sqrtXtr);

                        static constexpr auto kSqrtPi = koliop::portability::MathematicalConstant<ComputeFloatType>::kSqrtPi;

                        J_vec[1] = kSqrtPi * erfc_sqrt_Xtr;
                        J_vec[0] = 2.0 * (sqrtXtr_inv_on_exp_Xtr - J_vec[1]);

                        for(OrdinalType ind_k = 2; ind_k < J_vec_size; ++ind_k) {
                            J_vec[ind_k] = sqrtXtr_inv_on_exp_Xtr * Kokkos::abs(Xtr_pows[ind_k - 1]);
                        }

                        for(OrdinalType ind_k = 2; ind_k < J_vec_size; ++ind_k) {
                            J_vec[ind_k] += J_vec[ind_k - 1] * (static_cast<ValueType>(ind_k) - 1.5);
                        }

                        // Assumes I_mat is zeroed.
                        // I_mat(:) = 0.0;

                        for(OrdinalType ind_i = 0; ind_i < Xtr_pows_size; ++ind_i) {
                            for(OrdinalType ind_k = 0; ind_k <= ind_i; ++ind_k) {
                                // The numbers below represent:
                                // NORM2(a - r) / NORM2(r)
                                // MAXVAL(ABS(a - r))
                                // MAXVAL(ABS(a - r) / ABS(r))
                                // With a and r being approximation and reference.
                                // the reference is the Fortran code.
                                // Below we see multiple error report depending on
                                // the computation leading to the I_mat values. For
                                // each measurement, the same code transformation
                                // was applied to the fortran code so in theory we
                                // should have the same results. This is not the
                                // case as show and add even more mystery to what
                                // was reported in the comment at the tome of this
                                // function.
                                //
                                //
                                // GOOD:
                                // 9.0055254553295181E-16
                                // 6.39488462184090167E-14
                                // 1.66911273006078298E-12
                                // I_mat[ind_i] = I_mat[ind_i] * kCombMat[ind_i][ind_k];
                                //
                                // GOOD:
                                // 9.22848380287682281E-16
                                // 7.46069872548105195E-14
                                // 7.23852371101561738E-13
                                // I_mat[ind_i] = I_mat[ind_i] * J_vec[ind_k + (-offset_k + 1)];
                                //
                                // GOOD:
                                // 4.58332122622586803E-15
                                // 3120562176.
                                // 1.930299460045218E-12
                                // I_mat[ind_i] = I_mat[ind_i] * Xtr_pows[ind_i - ind_k];
                                //
                                // GOOD:
                                // 1.4048455258283236E-13
                                // 4.85505554119331165E+134
                                // 4.77752425883472662E-13
                                // I_mat[ind_i] = I_mat[ind_i] + exp_Xtr;
                                //
                                // GOOD:
                                // 9.26019275821356788E-16
                                // 7.10542735760100186E-14
                                // 7.23852371101561738E-13
                                // I_mat[ind_i] = I_mat[ind_i] *
                                //                kCombMat[ind_i][ind_k] * J_vec[ind_k + (-offset_k + 1)];
                                //
                                // MEH:
                                // +
                                // 2.39600218077008314E-15
                                // 4.17232513427734375E-7
                                // 1.32255877974644926E-11
                                // *
                                // 4.5951505930168964E-15
                                // 309237645312.
                                // 1.93029924759351626E-12
                                // I_mat[ind_i] = I_mat[ind_i] *
                                //                kCombMat[ind_i][ind_k] * Xtr_pows[ind_i - ind_k];
                                //
                                // MEH:
                                // 1.39915089874609498E-13
                                // 3.7122720000644766E+134
                                // 1.2319135169780248E-11
                                // I_mat[ind_i] = I_mat[ind_i] +
                                //                kCombMat[ind_i][ind_k] * exp_Xtr;
                                //
                                // GOOD:
                                // 9.27452285782984381E-16
                                // 7.10542735760100186E-14
                                // 7.67019556279941316E-13
                                // I_mat[ind_i] = I_mat[ind_i] +
                                //                J_vec[ind_k + (-offset_k + 1)] * Xtr_pows[ind_i - ind_k];
                                //
                                // BAD:
                                // 1.90019131286443E-11
                                // 7.09558662492781878E-7
                                // 5.82693268469331832E-7
                                // I_mat[ind_i] = I_mat[ind_i] +
                                //                J_vec[ind_k + (-offset_k + 1)] *
                                //                    Xtr_pows[ind_i - ind_k] * exp_Xtr;
                                //
                                // MEH:
                                // 1.39915089874609498E-13
                                // 3.7122720000644766E+134
                                // 1.2319135169780248E-11
                                // I_mat[ind_i] = I_mat[ind_i] +
                                //                kCombMat[ind_i][ind_k] * exp_Xtr;
                                //
                                // GOOD:
                                // 9.24565888941454475E-16
                                // 6.75015598972095177E-14
                                // 6.51097853232258926E-13
                                // I_mat[ind_i] = I_mat[ind_i] +
                                //                kCombMat[ind_i][ind_k] * J_vec[ind_k + (-offset_k + 1)] *
                                //                    Xtr_pows[ind_i - ind_k];
                                //
                                // BAD:
                                // 7.08135327836351362E-9
                                // 7.12772024513697033E-7
                                // 4.18202253096459523E-8
                                // I_mat[ind_i] = I_mat[ind_i] +
                                //                kCombMat[ind_i][ind_k] * J_vec[ind_k + (-offset_k + 1)] *
                                //                    Xtr_pows[ind_i - ind_k] * exp_Xtr;

                                // ind_i in [0, Constant::kIndexMax + 2 + 1)
                                I_mat[ind_i] = I_mat[ind_i] +
                                               kCombMat[ind_i][ind_k] * J_vec[ind_k + (-offset_k + 1)] *
                                                   Xtr_pows[ind_i - ind_k] * exp_Xtr;
                            }
                        }
                    }
                }

                /// Computation of L(n,i, sp_a, sp_b) with
                ///   L(0,i, sp_a, sp_b) = <D_dab / vTa**2 | u^i>
                ///   L(1,i, sp_a, sp_b) = <1/v d(D_dab)/dv | u^i>
                ///   L(2,i, sp_a, sp_b) = <nu_vab - nu_dab | u^i>
                ///   L(3,i, sp_a, sp_b) = <0.5 * v * d(nu_vab - nu_dab)/dv | u^i>
                ///   L(4,i, sp_a, sp_b) = <1 / v * d(nu_vab - nu_dab)/dv | u^i>
                /// Output: L_mat(0:4, 0:index_max_+1, 0:Nvpar_, 0:Nspecies_-1)
                ///
                template <typename ValueType,
                          typename View1D0,
                          typename View1D1,
                          typename Array2D>
                KOKKOS_INLINE_FUNCTION void
                DoComputeL(OrdinalType    species_extent,
                           OrdinalType    species_index,
                           bool           collision_interspecies,
                           const View1D0& hat_As,
                           const View1D0& hat_Zs,
                           const View1D1& local_n_s,
                           const View1D1& local_T_s,
                           ValueType      nu,
                           ValueType      vpar2_on_Ta,
                           Array2D        L_mat) {

                    const ValueType hat_Aa = hat_As(species_index);
                    const ValueType hat_Za = hat_Zs(species_index);

                    // Assumes L_mat is zeroed.
                    // L_mat(:,:) = 0.0;

                    for(OrdinalType ispeciesB = 0; ispeciesB < species_extent; ++ispeciesB) {
                        if(!(collision_interspecies || (species_index == ispeciesB))) {
                            continue;
                        }

                        const ValueType hat_Ab = hat_As(ispeciesB);
                        const ValueType hat_Zb = hat_Zs(ispeciesB);

                        static constexpr auto kSqrtPi = koliop::portability::MathematicalConstant<ValueType>::kSqrtPi;
                        static constexpr auto kSqrt2  = koliop::portability::MathematicalConstant<ValueType>::kSqrt2;
                        static constexpr auto kPi     = koliop::portability::MathematicalConstant<ValueType>::kPi;

                        const ValueType coeff_L = (0.75 * kSqrtPi * kSqrt2) *
                                                  nu * (hat_Zb * hat_Zb * local_n_s(ispeciesB)) / (hat_Za * hat_Za * local_n_s(species_index));

                        const ValueType vTb_on_vta_2 = (hat_Aa * local_T_s(ispeciesB)) / (hat_Ab * local_T_s(species_index));

                        {
                            const ValueType Xtr         = vpar2_on_Ta * 0.5 + vTb_on_vta_2 * (9.0 * kPi / 16.0);
                            const ValueType sqrtXtr     = Kokkos::sqrt(Xtr);
                            const ValueType sqrtXtr_inv = 1.0 / sqrtXtr;

                            ValueType I_mat[Constant::kIndexMax + 1 + 1]{};

                            DoComputeI(0, Xtr, sqrtXtr, sqrtXtr_inv, I_mat);

                            for(OrdinalType ind_i = 0; ind_i <= Constant::kIndexMax; ++ind_i) {
                                L_mat[0][ind_i] += coeff_L * I_mat[ind_i];
                            }

                            L_mat[1][0] -= coeff_L * sqrtXtr_inv;
                        }
                        {
                            const ValueType Xtr         = vpar2_on_Ta * 0.5 + vTb_on_vta_2 * 2.1;
                            const ValueType sqrtXtr     = Kokkos::sqrt(Xtr);
                            const ValueType sqrtXtr_inv = 1.0 / sqrtXtr;

                            ValueType I_mat[Constant::kIndexMax + 1 + 1]{};

                            DoComputeI(1, Xtr, sqrtXtr, sqrtXtr_inv, I_mat);

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

                    L_mat[3][0] = L_mat[3][0] +
                                  L_mat[2][0] * vpar2_on_Ta * 0.5 +
                                  L_mat[2][1] -
                                  L_mat[2][0];

                    for(OrdinalType ind_i = 1; ind_i <= Constant::kIndexMax; ++ind_i) {
                        L_mat[3][ind_i] = L_mat[3][ind_i] -
                                          L_mat[2][ind_i - 1] * static_cast<ValueType>(ind_i) * vpar2_on_Ta * 0.5 +
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

            /// Computation of the N_i,a ^ jl = sum_b M_i,ab ^ jl
            ///
            template <typename Najl,
                      typename ExecutionSpace,
                      typename View6D,
                      typename View5D,
                      typename View4D>
            static inline void
            DoComputeNajl(const Najl&    najl_provider,
                          ExecutionSpace an_execution_space,
                          OrdinalType    block_first_tor1,
                          OrdinalType    block_first_tor2,
                          OrdinalType    block_first_tor3,
                          OrdinalType    species_index,
                          const View4D&  n_s,
                          const View4D&  T_s,
                          const View5D&  nu_ab,
                          View6D&&       najl) {

                // NOTE: We use ValueType due to an hypothetical future Kokkos
                // release that would enable Kokkos::exp() on KokkosBatched::Vector.

                using ValueType = typename std::remove_reference_t<View6D>::value_type;
                // using ValueType = long double;

                const auto& vpar_extent    = najl_provider.vpar_extent_;
                const auto& species_extent = najl_provider.species_extent_;

                const auto& hat_As = najl_provider.hat_As_;
                const auto& hat_Zs = najl_provider.hat_Zs_;
                const auto& vparg  = najl_provider.vparg_;

                const auto& C_jl = najl_provider.C_jl_;

                const auto& collision_interspecies = najl_provider.collision_interspecies_;

                const auto kernel = KOKKOS_LAMBDA(OrdinalType ivpar, OrdinalType itor1, OrdinalType itor2, OrdinalType itor3) {

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType block_local_index_itor1 = itor1 - block_first_tor1;
                    const OrdinalType block_local_index_itor2 = itor2 - block_first_tor2;
                    const OrdinalType block_local_index_itor3 = itor3 - block_first_tor3;

                    const OffsetType flattened_index = Najl::CvCdDriverType::kBlockExtentTor1 * Najl::CvCdDriverType::kBlockExtentTor2 * block_local_index_itor3 +
                                                       Najl::CvCdDriverType::kBlockExtentTor1 * block_local_index_itor2 +
                                                       block_local_index_itor1;

                    const OrdinalType batch_element_index = (flattened_index) % Najl::CvCdDriverType::BatchTraitsType::kBatchSize;
                    const OrdinalType batch_index         = (flattened_index / (Najl::CvCdDriverType::BatchTraitsType::kBatchSize)) /* % Najl::CvCdDriverType::kBatchCount */;

                    auto local_N0 = Kokkos::subview(najl,
                                                    batch_index,
                                                    ivpar,
                                                    0,
                                                    Kokkos::ALL(), // j
                                                    Kokkos::ALL(), // l
                                                    batch_element_index);

                    auto local_N1 = Kokkos::subview(najl,
                                                    batch_index,
                                                    ivpar,
                                                    1,
                                                    Kokkos::ALL(), // j
                                                    Kokkos::ALL(), // l
                                                    batch_element_index);

                    auto local_N2 = Kokkos::subview(najl,
                                                    batch_index,
                                                    ivpar,
                                                    2,
                                                    Kokkos::ALL(), // j
                                                    Kokkos::ALL(), // l
                                                    batch_element_index);

                    const auto local_n_s = Kokkos::subview(n_s,
                                                           itor1,
                                                           itor2,
                                                           itor3,
                                                           Kokkos::ALL());

                    const auto local_T_s = Kokkos::subview(T_s,
                                                           itor1,
                                                           itor2,
                                                           itor3,
                                                           Kokkos::ALL());

                    // NOTE: The memset(0) of the local_Nn used to be done in
                    // Najl::Compute via a deep_copy.
                    for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                        for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                            local_N0(l, j) = static_cast<ValueType>(0.0);
                            local_N1(l, j) = static_cast<ValueType>(0.0);
                            local_N2(l, j) = static_cast<ValueType>(0.0);
                        }
                    }

                    const ValueType vpar2_on_Ta = vparg(ivpar) * vparg(ivpar) / local_T_s(species_index);

                    // NOTE: We could reduce the size of L_mat by using some
                    // condition. For instance:
                    // (ind_i + 1) == (Constant::kIndexMax + 1) ?
                    //      0.0 :
                    //      L_mat[N][ind_i + 1]
                    // NOTE: Careful, L_mat is layout right
                    ValueType L_mat[5][Constant::kIndexMax + 1 + 1]{};

                    DoComputeL<ValueType>(species_extent,
                                          species_index,
                                          collision_interspecies,
                                          hat_As, hat_Zs,
                                          local_n_s, local_T_s,
                                          nu_ab(itor1, itor2, itor3, species_index, species_index),
                                          vpar2_on_Ta,
                                          L_mat);

                    for(OrdinalType ind_i = 0; ind_i <= Constant::kIndexMax; ++ind_i) {
                        ValueType N_coeff = L_mat[1][ind_i] * 2.0 +
                                            L_mat[2][ind_i] * (3.0 - vpar2_on_Ta) +
                                            L_mat[4][ind_i] * (2.0 + vpar2_on_Ta);

                        if(ind_i != Constant::kIndexMax) {
                            for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                                for(OrdinalType j = 1; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                    local_N0(l, j) = local_N0(l, j) -
                                                     static_cast<ValueType>(j) * N_coeff * C_jl(ind_i, j - 1, l);
                                }
                            }
                        }

                        N_coeff = N_coeff -
                                  2.0 * L_mat[0][ind_i] -
                                  2.0 * L_mat[2][ind_i + 1];

                        for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                            for(OrdinalType j = 1; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                local_N0(l, j) += static_cast<ValueType>(j) * N_coeff * C_jl(ind_i, j, l);
                            }
                        }

                        N_coeff = -L_mat[0][ind_i] +
                                  L_mat[1][ind_i] -
                                  L_mat[2][ind_i + 1] +
                                  L_mat[2][ind_i] * (2.0 - vpar2_on_Ta * 0.5) +
                                  L_mat[3][ind_i];

                        if(ind_i != Constant::kIndexMax) {
                            for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                                local_N1(l, 0) += vparg(ivpar) * C_jl(ind_i, 0, l) * N_coeff;
                                for(OrdinalType j = 1; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                    local_N1(l, j) += vparg(ivpar) * (C_jl(ind_i, j, l) * N_coeff +
                                                                      2.0 * static_cast<ValueType>(j) * L_mat[2][ind_i] * (C_jl(ind_i, j, l) - C_jl(ind_i, j - 1, l)));
                                }
                            }
                        } else {
                            for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                                for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                    local_N1(l, j) += vparg(ivpar) *
                                                      C_jl(Constant::kIndexMax, j, l) *
                                                      (N_coeff +
                                                       2.0 * static_cast<ValueType>(j) * L_mat[2][Constant::kIndexMax]);
                                }
                            }
                        }

                        N_coeff = local_T_s(species_index) * (L_mat[0][ind_i] +
                                                              L_mat[2][ind_i] * vpar2_on_Ta * 0.5);

                        for(OrdinalType l = 0; l < Constant::kMaximumLaguerrePolynomialDegree; ++l) {
                            for(OrdinalType j = 0; j < Constant::kMaximumLaguerrePolynomialDegree; ++j) {
                                local_N2(l, j) += C_jl(ind_i, j, l) * N_coeff;
                            }
                        }
                    }
                };

                const auto a_policy = Kokkos::MDRangePolicy<
                    // Kokkos::LaunchBounds<256, 1>, Kokkos::IndexType<OrdinalType>,
                    ExecutionSpace,
                    Kokkos::Rank<4,
                                 Kokkos::Iterate::Left,
                                 Kokkos::Iterate::Left>>{an_execution_space,
                                                         {0,
                                                          block_first_tor1,
                                                          block_first_tor2,
                                                          block_first_tor3},
                                                         {vpar_extent,
                                                          block_first_tor1 + Najl::CvCdDriverType::kBlockExtentTor1,
                                                          block_first_tor2 + Najl::CvCdDriverType::kBlockExtentTor2,
                                                          block_first_tor3 + Najl::CvCdDriverType::kBlockExtentTor3}};

                Kokkos::parallel_for("DoComputeNajl",
                                     a_policy, kernel);
            }

        } // namespace detail

        ////////////////////////////////////////////////////////////////////////
        // Najl definition
        ////////////////////////////////////////////////////////////////////////

        /// Computes the Najl coefficients.
        /// The N0 coefficients are stored in a
        /// ScalarBlockTridiagonalByBlockMatrixType. See
        /// BatchedBlockTridiagonalByBlockTRF for the layout. Note that the Najl
        /// coefficients are accessed by selecting the proper diagonal of the
        /// matrix representation.
        ///
        /// For instance:
        /// ScalarMatrixType{}(5, 4, 2, 2, 1, 0)
        /// represents the first (0) batch element l=1 and j=2
        /// of the N2 (2) coefficient block at the fifth vpar
        /// (4) of the sixth batch (5).
        /// And similary:
        /// ScalarMatrixType{}(5, 4, 1, 2, 1, 0) is the same thing but for N1.
        /// ScalarMatrixType{}(5, 4, 0, 2, 1, 0) is the same thing but for N0.
        ///
        template <typename CvCdDriver>
        class Najl {
        public:
            using CvCdDriverType = CvCdDriver;

            /// The Najl coefficients are stored in a data structure similar to
            /// how the one used for the matrix in Operator::Apply().
            ///
            using MatrixKindType = typename CvCdDriverType::MatrixKindType;

        public:
            Najl(OrdinalType                                                vpar_extent,
                 OrdinalType                                                species_extent,
                 bool                                                       collision_interspecies,
                 const koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                 const koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                 const koliop::utility::View::InternalInputL1<const Flo64>& vparg);

            template <typename ExecutionSpace>
            void Compute(ExecutionSpace                                       an_execution_space,
                         OrdinalType                                          block_first_tor1,
                         OrdinalType                                          block_first_tor2,
                         OrdinalType                                          block_first_tor3,
                         OrdinalType                                          species_index,
                         const koliop::utility::View::MDL4<ComputeFloatType>& n_s,
                         const koliop::utility::View::MDL4<ComputeFloatType>& T_s,
                         const koliop::utility::View::MDL5<ComputeFloatType>& nu_ab,
                         typename MatrixKindType::ScalarType&                 Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            // Constant, externally computed values

            OrdinalType vpar_extent_;
            OrdinalType species_extent_;
            bool        collision_interspecies_;

            koliop::utility::View::InternalInputL1<const Flo64> hat_As_;
            koliop::utility::View::InternalInputL1<const Flo64> hat_Zs_;
            koliop::utility::View::InternalInputL1<const Flo64> vparg_;

            // Constant, internally computed values

            koliop::utility::View::MDL<ComputeFloatType[2 * Constant::kMaximumLaguerrePolynomialDegree - 1]
                                                       [Constant::kMaximumLaguerrePolynomialDegree]
                                                       [Constant::kMaximumLaguerrePolynomialDegree]>
                C_jl_;
        };


        ////////////////////////////////////////////////////////////////////////
        // Najl methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename CvCdDriver>
        Najl<CvCdDriver>::Najl(OrdinalType                                                vpar_extent,
                               OrdinalType                                                species_extent,
                               bool                                                       collision_interspecies,
                               const koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                               const koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                               const koliop::utility::View::InternalInputL1<const Flo64>& vparg)
            : vpar_extent_{vpar_extent}
            , species_extent_{species_extent}
            , collision_interspecies_{collision_interspecies}
            , hat_As_{hat_As}
            , hat_Zs_{hat_Zs}
            , vparg_{vparg}

            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_")} {

            // TODO(Etienne M): Observe how the C_j_ and C_jl_ values are
            // constant. They could be compile time computed or hardcoded.
            detail::DoPrepareCjl(C_jl_);
        }

        template <typename CvCdDriver>
        template <typename ExecutionSpace>
        void Najl<CvCdDriver>::Compute(ExecutionSpace                                       an_execution_space,
                                       OrdinalType                                          block_first_tor1,
                                       OrdinalType                                          block_first_tor2,
                                       OrdinalType                                          block_first_tor3,
                                       OrdinalType                                          species_index,
                                       const koliop::utility::View::MDL4<ComputeFloatType>& n_s,
                                       const koliop::utility::View::MDL4<ComputeFloatType>& T_s,
                                       const koliop::utility::View::MDL5<ComputeFloatType>& nu_ab,
                                       typename MatrixKindType::ScalarType&                 Najl) const {

            KOLIOP_ASSERT(Najl.size() != 0);

            // // NOTE: This is not done in the kernel what is the perf of such
            // // memset. See if putting that in the main kernel improves the
            // // duration.
            // // NOTE: there is a bug in Kokkos 4.5.1 that does not zero every
            // // thing! See, zeromemset: https://github.com/kokkos/kokkos/pull/7797/files#diff-4fcf172cdeb761406dda3813e1f69edf12e5df844702ca4a3c2c025b1478e6d5
            // Kokkos::deep_copy(an_execution_space, Najl, static_cast<ComputeFloatType>(0.0));

            // TODO(Etienne M): It is not currently possible to do something
            // like:
            // Kokkos::exp(T{}) where T is a KokkosBatched::Vector
            // For this reason, we work on the scalar version of the view.

            // typename MatrixKindType::VectorType Najl_view{reinterpret_cast<typename CvCdDriverType::BatchTraitsType::ThreadVectorType*>(Najl.data()),
            //                                                   static_cast<std::size_t>(CvCdDriverType::kBatchCount),
            //                                                   static_cast<std::size_t>(vpar_extent_)};

            typename MatrixKindType::ScalarType Najl_view{Najl.data(),
                                                          static_cast<std::size_t>(CvCdDriverType::kBatchCount),
                                                          static_cast<std::size_t>(vpar_extent_)};

            detail::DoComputeNajl(*this,
                                  an_execution_space,
                                  block_first_tor1,
                                  block_first_tor2,
                                  block_first_tor3,
                                  species_index,
                                  n_s,
                                  T_s,
                                  nu_ab,
                                  Najl_view);
        }

    } // namespace collision
} // namespace koliop

#endif
