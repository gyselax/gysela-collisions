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

                const Constant::DeviceExecutionSpaceType an_execution_space{};

                koliop::utility::Copy::Shallow(an_execution_space,
                                               a_host_visible_C_jl,
                                               C_jl);

                an_execution_space.fence();
            }

            namespace {
                // NOTE: I use an unamed namespace because it is unknown to me if
                // adding static prior KOKKOS_INLINE_FUNCTION is legal.

                template <typename ValueType,
                          typename Array1D>
                KOKKOS_INLINE_FUNCTION void
                DoComputeI(OffsetType offset,
                           ValueType  Xtr,
                           ValueType  sqrtXtr,
                           ValueType  sqrtXtr_inv,
                           Array1D    I_mat) {

                    // NOTE: Numerical accuracy of this whole function isn't
                    // good!
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

                    static constexpr OrdinalType the_Xtr_pows_size = Constant::kIndexMax + 1 + 1;
                    static constexpr OrdinalType the_J_vec_size    = Constant::kIndexMax + 2 + 1;

                    // Precompute kCombMat(n, k): n! /(k! * (n-k)!)
                    // AKA: binomial coeff/pascal's triangle
                    // n are the ind_i
                    // k are the ind_k
                    // clang-format off
                    static constexpr ValueType kCombMat[the_Xtr_pows_size][the_Xtr_pows_size] =
                        {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                         {1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                         {1.0, 2.0, 1.0, 0.0, 0.0, 0.0},
                         {1.0, 3.0, 3.0, 1.0, 0.0, 0.0},
                         {1.0, 4.0, 6.0, 4.0, 1.0, 0.0},
                         {1.0, 5.0, 10.0, 10.0, 5.0, 1.0}};
                    // clang-format on

                    ValueType Xtr_pows[the_Xtr_pows_size]; // std::pow(-Xtr, i);
                    ValueType J_vec[the_J_vec_size];

                    // for(OrdinalType ind_i = 0; ind_i < the_Xtr_pows_size; ++ind_i) {
                    //     Xtr_pows[ind_i] = Kokkos::pow(-Xtr, static_cast<ValueType>(ind_i));
                    // }

                    Xtr_pows[0] = 1.0;

                    for(OrdinalType ind_i = 1; ind_i < the_Xtr_pows_size; ++ind_i) {
                        Xtr_pows[ind_i] = -Xtr * Xtr_pows[ind_i - 1];
                    }

                    const ValueType the_exp_Xtr                = Kokkos::exp(Xtr);
                    const ValueType the_sqrtXtr_inv_on_exp_Xtr = sqrtXtr_inv / the_exp_Xtr;
                    const ValueType the_erfc_sqrt_Xtr          = Kokkos::erfc(sqrtXtr);

                    J_vec[1] = koliop::portability::MathematicalConstant<ValueType>::kSqrtPi * the_erfc_sqrt_Xtr;
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
                            // I_mat[ind_i] = I_mat[ind_i] * J_vec[ind_k + offset];
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
                            // I_mat[ind_i] = I_mat[ind_i] + the_exp_Xtr;
                            //
                            // GOOD:
                            // 9.26019275821356788E-16
                            // 7.10542735760100186E-14
                            // 7.23852371101561738E-13
                            // I_mat[ind_i] = I_mat[ind_i] *
                            //                kCombMat[ind_i][ind_k] * J_vec[ind_k + offset];
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
                            //                kCombMat[ind_i][ind_k] * the_exp_Xtr;
                            //
                            // GOOD:
                            // 9.27452285782984381E-16
                            // 7.10542735760100186E-14
                            // 7.67019556279941316E-13
                            // I_mat[ind_i] = I_mat[ind_i] +
                            //                J_vec[ind_k + offset] * Xtr_pows[ind_i - ind_k];
                            //
                            // BAD:
                            // 1.90019131286443E-11
                            // 7.09558662492781878E-7
                            // 5.82693268469331832E-7
                            // I_mat[ind_i] = I_mat[ind_i] +
                            //                J_vec[ind_k + offset] *
                            //                    Xtr_pows[ind_i - ind_k] * the_exp_Xtr;
                            //
                            // MEH:
                            // 1.39915089874609498E-13
                            // 3.7122720000644766E+134
                            // 1.2319135169780248E-11
                            // I_mat[ind_i] = I_mat[ind_i] +
                            //                kCombMat[ind_i][ind_k] * the_exp_Xtr;
                            //
                            // GOOD:
                            // 9.24565888941454475E-16
                            // 6.75015598972095177E-14
                            // 6.51097853232258926E-13
                            // I_mat[ind_i] = I_mat[ind_i] +
                            //                kCombMat[ind_i][ind_k] * J_vec[ind_k + offset] *
                            //                    Xtr_pows[ind_i - ind_k];
                            //
                            // BAD:
                            // 7.08135327836351362E-9
                            // 7.12772024513697033E-7
                            // 4.18202253096459523E-8
                            // I_mat[ind_i] = I_mat[ind_i] +
                            //                kCombMat[ind_i][ind_k] * J_vec[ind_k + offset] *
                            //                    Xtr_pows[ind_i - ind_k] * the_exp_Xtr;

                            // ind_i in [0, Constant::kIndexMax + 2 + 1)
                            I_mat[ind_i] = I_mat[ind_i] +
                                           kCombMat[ind_i][ind_k] * J_vec[ind_k + offset] *
                                               Xtr_pows[ind_i - ind_k] * the_exp_Xtr;
                        }
                    }
                }

                template <typename ValueType,
                          typename View1D0,
                          typename View1D1,
                          typename Array2D>
                KOKKOS_INLINE_FUNCTION void
                DoComputeL(OrdinalType    the_species_extent,
                           OrdinalType    pSpecies_id,
                           bool           collision_interspecies,
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
                        if(!(collision_interspecies || (pSpecies_id == ispeciesB))) {
                            continue;
                        }

                        const ValueType hat_Ab = hat_As(ispeciesB);
                        const ValueType hat_Zb = hat_Zs(ispeciesB);

                        const ValueType coeff_L = (0.75 * koliop::portability::MathematicalConstant<ValueType>::kSqrtPi * koliop::portability::MathematicalConstant<ValueType>::kSqrt2) *
                                                  nu * (hat_Zb * hat_Zb * local_n_s(ispeciesB)) / (hat_Za * hat_Za * local_n_s(pSpecies_id));

                        const ValueType vTb_on_vta_2 = (hat_Aa * local_T_s(ispeciesB)) / (hat_Ab * local_T_s(pSpecies_id));

                        {
                            const ValueType Xtr         = vpar2_on_Ta * 0.5 + vTb_on_vta_2 * (9.0 * koliop::portability::MathematicalConstant<ValueType>::kPi / 16.0);
                            const ValueType sqrtXtr     = Kokkos::sqrt(Xtr);
                            const ValueType sqrtXtr_inv = 1.0 / sqrtXtr;

                            ValueType I_mat[Constant::kIndexMax + 1 + 1]{};

                            DoComputeI(1, Xtr, sqrtXtr, sqrtXtr_inv, I_mat);

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

                            DoComputeI(0, Xtr, sqrtXtr, sqrtXtr_inv, I_mat);

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

            template <typename Najl,
                      typename ExecutionSpace,
                      typename View6D,
                      typename View5D,
                      typename View4D>
            static inline void
            DoComputeNajl(const Najl&    the_najl_provider,
                          ExecutionSpace an_execution_space,
                          OrdinalType    the_block_first_r,
                          OrdinalType    the_block_first_theta,
                          OrdinalType    the_block_first_phi,
                          const View4D&  n_s,
                          const View4D&  T_s,
                          const View5D&  nu_ab,
                          View6D&&       the_Najl) {

                // NOTE: We use ValueType due to an hypothetical future Kokkos
                // release that would enable Kokkos::exp() on KokkosBatched::Vector.

                using ValueType = typename std::remove_reference_t<View6D>::value_type;
                // using ValueType = long double;

                const auto& the_vpar_extent    = the_najl_provider.the_vpar_extent_;
                const auto& the_species_extent = the_najl_provider.the_species_extent_;

                const auto& hat_As = the_najl_provider.hat_As_;
                const auto& hat_Zs = the_najl_provider.hat_Zs_;
                const auto& vparg  = the_najl_provider.vparg_;

                const auto& C_jl = the_najl_provider.C_jl_;

                const auto& pSpecies_id            = the_najl_provider.pSpecies_id_;
                const auto& collision_interspecies = the_najl_provider.collision_interspecies_;

                const auto the_kernel = KOKKOS_LAMBDA(OrdinalType ivpar, OrdinalType ir, OrdinalType itheta, OrdinalType iphi) {

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType the_block_local_index_r     = ir - the_block_first_r;
                    const OrdinalType the_block_local_index_theta = itheta - the_block_first_theta;
                    const OrdinalType the_block_local_index_phi   = iphi - the_block_first_phi;

                    const OffsetType the_flattened_index = Najl::CvCdDriverType::kBlockExtentR * Najl::CvCdDriverType::kBlockExtentTheta * the_block_local_index_phi +
                                                           Najl::CvCdDriverType::kBlockExtentR * the_block_local_index_theta +
                                                           the_block_local_index_r;

                    const OrdinalType the_batch_element_index = (the_flattened_index) % Najl::CvCdDriverType::BatchTraitsType::kBatchSize;
                    const OrdinalType the_batch_index         = (the_flattened_index / (Najl::CvCdDriverType::BatchTraitsType::kBatchSize)) /* % Najl::CvCdDriverType::kBatchCount */;

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

                    // NOTE: We could reduce the size of L_mat by using some
                    // condition. For instance:
                    // (ind_i + 1) == (Constant::kIndexMax + 1) ?
                    //      0.0 :
                    //      L_mat[N][ind_i + 1]
                    // NOTE: Careful, L_mat is layout right
                    ValueType L_mat[5][Constant::kIndexMax + 1 + 1]{};

                    DoComputeL<ValueType>(the_species_extent,
                                          pSpecies_id,
                                          collision_interspecies,
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
                                    local_N0(l, j) -= static_cast<ValueType>(j) * N_coeff * C_jl(ind_i, j - 1, l);
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

                        N_coeff = local_T_s(pSpecies_id) * (L_mat[0][ind_i] +
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
                                                          the_block_first_r,
                                                          the_block_first_theta,
                                                          the_block_first_phi},
                                                         {the_vpar_extent,
                                                          the_block_first_r + Najl::CvCdDriverType::kBlockExtentR,
                                                          the_block_first_theta + Najl::CvCdDriverType::kBlockExtentTheta,
                                                          the_block_first_phi + Najl::CvCdDriverType::kBlockExtentPhi}};

                Kokkos::parallel_for("DoComputeNajl",
                                     a_policy, the_kernel);
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
            Najl(OrdinalType                                                pSpecies_id,
                 OrdinalType                                                the_vpar_extent,
                 OrdinalType                                                the_species_extent,
                 bool                                                       collision_interspecies,
                 const koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                 const koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                 const koliop::utility::View::InternalInputL1<const Flo64>& vparg);

            template <typename ExecutionSpace>
            void Compute(ExecutionSpace                                       an_execution_space,
                         OrdinalType                                          the_block_first_r,
                         OrdinalType                                          the_block_first_theta,
                         OrdinalType                                          the_block_first_phi,
                         const koliop::utility::View::MDL4<ComputeFloatType>& n_s,
                         const koliop::utility::View::MDL4<ComputeFloatType>& T_s,
                         const koliop::utility::View::MDL5<ComputeFloatType>& nu_ab,
                         typename MatrixKindType::ScalarType&                 the_Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            // Constant, externally computed values

            OrdinalType pSpecies_id_;
            OrdinalType the_vpar_extent_;
            OrdinalType the_species_extent_;
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
        // CvCdDriver definition
        ////////////////////////////////////////////////////////////////////////

        template <typename CvCdDriver>
        Najl<CvCdDriver>::Najl(OrdinalType                                                pSpecies_id,
                               OrdinalType                                                the_vpar_extent,
                               OrdinalType                                                the_species_extent,
                               bool                                                       collision_interspecies,
                               const koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                               const koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                               const koliop::utility::View::InternalInputL1<const Flo64>& vparg)
            : pSpecies_id_{pSpecies_id}
            , the_vpar_extent_{the_vpar_extent}
            , the_species_extent_{the_species_extent}
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
                                       OrdinalType                                          the_block_first_r,
                                       OrdinalType                                          the_block_first_theta,
                                       OrdinalType                                          the_block_first_phi,
                                       const koliop::utility::View::MDL4<ComputeFloatType>& n_s,
                                       const koliop::utility::View::MDL4<ComputeFloatType>& T_s,
                                       const koliop::utility::View::MDL5<ComputeFloatType>& nu_ab,
                                       typename MatrixKindType::ScalarType&                 the_Najl) const {

            KOLIOP_ASSERT(the_Najl.size() != 0);

            // TODO(Etienne M): what is the perf of such memset.
            // See if putting that in the main kernel improves the duration.
            Kokkos::deep_copy(an_execution_space, the_Najl, static_cast<ComputeFloatType>(0.0));

            // TODO(Etienne M): It is not currently possible to do something
            // like:
            // Kokkos::exp(T{}) where T is a KokkosBatched::Vector
            // For this reason, we work on the scalar version of the view.

            // typename MatrixKindType::VectorType the_Najl_view{reinterpret_cast<typename CvCdDriverType::BatchTraitsType::ThreadVectorType*>(the_Najl.data()),
            //                                                   static_cast<std::size_t>(CvCdDriverType::kBatchCount),
            //                                                   static_cast<std::size_t>(the_vpar_extent_)};

            typename MatrixKindType::ScalarType the_Najl_view{the_Najl.data(),
                                                              static_cast<std::size_t>(CvCdDriverType::kBatchCount),
                                                              static_cast<std::size_t>(the_vpar_extent_)};

            detail::DoComputeNajl(*this,
                                  an_execution_space,
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

#endif
