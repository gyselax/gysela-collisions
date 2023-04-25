// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        namespace {
            // NOTE: I use an unamed namespace because it is unknown to me if
            // adding static prior KOKKOS_INLINE_FUNCTION is legal.

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            DoComputeLaguerrePol(OrdinalType l, ComputeFloatType x) {
                // TODO(Etienne M): find a better way to compute that, based on l.
                switch(l) {
                    case 0:
                        return 1.0;
                    case 1:
                        return -x + 1.0;
                    case 2:
                        return (1.0 / 2.0) + (-4 + /* 1 * */ x) * x;
                    case 3:
                        // Horner -> dependency            : 1 + (-3 + (1.5 - 1/6 * x) * x) * x
                        // but FMA friendly.
                        // Naive  -> latency and dependency: (1.0 / 6.0) * (-(x * x * x) + 9.0 * x * x - 18.0 * x + 6.0);
                        // Slow.
                        // Factorized -> low latency       : ((-1.0 / 6.0) * (x - 0.41577455678347908331)) * ((x - 2.29428036027904171982) * (x - 6.28994508293747919686))
                        return 1.0 + (-3.0 + (1.5 - (1.0 / 6.0) * x) * x) * x;
                    default:
                        return -1.0;
                }
            }

        } // namespace

        template <typename View6D,
                  typename View4D,
                  typename View2D>
        static inline void
        DoComputeAlphaProjection(const Operator& an_operator,
                                 OrdinalType     block_first_r,
                                 OrdinalType     block_first_theta,
                                 OrdinalType     block_first_phi,
                                 const View2D&   mask_coll,
                                 const View6D&   f5d_s,
                                 View4D&&        the_rhs) {

            const auto& the_mu_extent   = an_operator.the_mu_extent_;
            const auto& the_vpar_extent = an_operator.the_vpar_extent_;

            const auto& mug          = an_operator.mug_;
            const auto& coeff_intdmu = an_operator.coeff_intdmu_;
            const auto& B_norm       = an_operator.B_norm_;
            const auto& T_s          = an_operator.T_s_;

            const auto& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const OrdinalType the_block_local_index_r     = ir % Operator::kBlockExtentR;
                const OrdinalType the_block_local_index_theta = itheta % Operator::kBlockExtentTheta;
                const OrdinalType the_block_local_index_phi   = iphi % Operator::kBlockExtentPhi;

                const OffsetType the_flattened_index = Operator::kBlockExtentR * Operator::kBlockExtentTheta * the_block_local_index_phi +
                                                       Operator::kBlockExtentR * the_block_local_index_theta +
                                                       the_block_local_index_r;

                const OrdinalType the_batch_index         = the_flattened_index / Operator::AlphaSolverType::kBatchSize;
                const OrdinalType the_batch_element_index = the_flattened_index % Operator::AlphaSolverType::kBatchSize;

                if(mask_coll(ir, itheta)) {
                    const ComputeFloatType BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                    for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(OrdinalType ipol = 0; ipol < Operator::kMaximumLaguerrePolynomialDegree; ++ipol) {
                            for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                                const ComputeFloatType muBoT     = mug(imu) * BoT;
                                const ComputeFloatType coeff_tot = DoComputeLaguerrePol(ipol, muBoT) * BoT * coeff_intdmu(imu) * Kokkos::exp(-muBoT);

                                the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) += coeff_tot *
                                                                                                  f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                            }
                        }
                    }
                } else {
                    for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(OrdinalType ipol = 0; ipol < Operator::kMaximumLaguerrePolynomialDegree; ++ipol) {
                            the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) = 0.0;
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{block_first_r,
                                                                                              block_first_theta,
                                                                                              block_first_phi},
                                                                                             {block_first_r + Operator::kBlockExtentR,
                                                                                              block_first_theta + Operator::kBlockExtentTheta,
                                                                                              block_first_phi + Operator::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeAlphaProjection",
                                 a_policy, the_kernel);
        }

        template <typename View2D0,
                  typename View2D1,
                  typename View1D>
        static inline void
        DoComputeBlockTo1D(const Operator& an_operator,
                           OrdinalType     block_first_r,
                           OrdinalType     block_first_theta,
                           OrdinalType     block_first_phi,
                           const View2D0&  mask_coll,
                           View2D1&&       n1d_s,
                           View2D1&&       T1d_s,
                           View1D&&        U_par1d,
                           View1D&&        q_par_on_nT1d,
                           View1D&&        nu1d) {

            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& n_s           = an_operator.n_s_;
            const auto& U_par_s       = an_operator.U_par_s_;
            const auto& T_s           = an_operator.T_s_;
            const auto& nu_ab         = an_operator.nu_ab_;
            const auto& q_par_on_nT_s = an_operator.q_par_on_nT_s_;

            const auto& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const OrdinalType the_block_local_index_r     = ir % Operator::kBlockExtentR;
                const OrdinalType the_block_local_index_theta = itheta % Operator::kBlockExtentTheta;
                const OrdinalType the_block_local_index_phi   = iphi % Operator::kBlockExtentPhi;

                const OffsetType the_flattened_index = Operator::kBlockExtentR * Operator::kBlockExtentTheta * the_block_local_index_phi +
                                                       Operator::kBlockExtentR * the_block_local_index_theta +
                                                       the_block_local_index_r;

                if(!mask_coll(ir, itheta)) {
                    return;
                }

                for(OrdinalType ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    n1d_s(the_flattened_index, ispecies) = n_s(ir, itheta, iphi, ispecies);
                    T1d_s(the_flattened_index, ispecies) = T_s(ir, itheta, iphi, ispecies);
                }

                U_par1d(the_flattened_index)       = U_par_s(ir, itheta, iphi, pSpecies_id);
                q_par_on_nT1d(the_flattened_index) = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);
                nu1d(the_flattened_index)          = nu_ab(ir, itheta, iphi, pSpecies_id, pSpecies_id);
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{block_first_r,
                                                                                              block_first_theta,
                                                                                              block_first_phi},
                                                                                             {block_first_r + Operator::kBlockExtentR,
                                                                                              block_first_theta + Operator::kBlockExtentTheta,
                                                                                              block_first_phi + Operator::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeBlockTo1D",
                                 a_policy, the_kernel);
        }

        template <typename View4D0,
                  typename View4D1,
                  typename View2D,
                  typename View1D>
        static inline void
        DoComputeCvCdSource(const Operator& an_operator,
                            const View2D&   T1d_s,
                            const View1D&   U_par1d,
                            const View1D&   q_par_on_nT1d,
                            const View4D0&  N0,
                            const View4D0&  N1,
                            const View4D0&  N2,
                            View4D1&&       the_rhs) {

            // TODO(Etienne M): The Nia's matrix indexing is transposed. Careful
            // with N(k, j, l) which is how it is stored and N(k, l, j) how it
            // is used when building the matrix T.

            const auto& the_vpar_extent = an_operator.the_vpar_extent_;

            const auto& vparg = an_operator.vparg_;

            const auto&            pSpecies_id = an_operator.pSpecies_id_;
            const ComputeFloatType dt          = an_operator.dt_;

            const auto the_kernel = KOKKOS_LAMBDA(auto the_batch_index, auto ivpar, auto ipol, auto the_batch_element_index) {
                const OrdinalType ispace = the_batch_index * Operator::AlphaSolverType::kBatchSize +
                                           the_batch_element_index;

                const ComputeFloatType vpar_tmp       = vparg(ivpar);
                const ComputeFloatType Ta             = T1d_s(ispace, pSpecies_id);
                const ComputeFloatType U_para         = U_par1d(ispace);
                const ComputeFloatType q_para_on_naTa = q_par_on_nT1d(ispace);

                // TODO(Etienne M): Can we simplify that ?

                the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) -= dt / (5.0 * Ta) *
                                                                                  ((5.0 * U_para - 3.0 * q_para_on_naTa) * (vpar_tmp * N0(ivpar, ispace, 0, ipol) + N1(ivpar, ispace, 0, ipol)) +
                                                                                   q_para_on_naTa * vpar_tmp * vpar_tmp / Ta * (vpar_tmp * N0(ivpar, ispace, 0, ipol) + 3.0 * N1(ivpar, ispace, 0, ipol)) +
                                                                                   2.0 * q_para_on_naTa * (3.0 * vpar_tmp / Ta * N2(ivpar, ispace, 0, ipol) - vpar_tmp * N0(ivpar, ispace, 1, ipol) - N1(ivpar, ispace, 1, ipol)));
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Right,
                                                                     Kokkos::Iterate::Right>>{{0,
                                                                                               0,
                                                                                               0,
                                                                                               0},
                                                                                              {Operator::kBatchCount,
                                                                                               the_vpar_extent,
                                                                                               Operator::kMaximumLaguerrePolynomialDegree,
                                                                                               Operator::AlphaSolverType::kBatchSize}};

            Kokkos::parallel_for("DoComputeCvCdSource",
                                 a_policy, the_kernel);
        }

        template <typename View6D,
                  typename View4D,
                  typename View2D>
        static inline void
        DoComputeReconstruction(const Operator& an_operator,
                                OrdinalType     block_first_r,
                                OrdinalType     block_first_theta,
                                OrdinalType     block_first_phi,
                                const View2D&   mask_coll,
                                View6D&&        f5d_s,
                                const View4D&   the_rhs) {

            const auto& the_mu_extent   = an_operator.the_mu_extent_;
            const auto& the_vpar_extent = an_operator.the_vpar_extent_;

            const auto& mug    = an_operator.mug_;
            const auto& B_norm = an_operator.B_norm_;
            const auto& T_s    = an_operator.T_s_;

            const auto& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const OrdinalType the_block_local_index_r     = ir % Operator::kBlockExtentR;
                const OrdinalType the_block_local_index_theta = itheta % Operator::kBlockExtentTheta;
                const OrdinalType the_block_local_index_phi   = iphi % Operator::kBlockExtentPhi;

                const OffsetType the_flattened_index = Operator::kBlockExtentR * Operator::kBlockExtentTheta * the_block_local_index_phi +
                                                       Operator::kBlockExtentR * the_block_local_index_theta +
                                                       the_block_local_index_r;

                const OrdinalType the_batch_index         = the_flattened_index / Operator::AlphaSolverType::kBatchSize;
                const OrdinalType the_batch_element_index = the_flattened_index % Operator::AlphaSolverType::kBatchSize;

                if(!mask_coll(ir, itheta)) {
                    return;
                }

                const ComputeFloatType BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                        const ComputeFloatType muBoT       = mug(imu) * BoT;
                        ComputeFloatType       f5d_s_value = 0.0;

                        for(OrdinalType ipol = 0; ipol < Operator::kMaximumLaguerrePolynomialDegree; ++ipol) {
                            f5d_s_value += the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) *
                                           DoComputeLaguerrePol(ipol, muBoT);
                        }

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = f5d_s_value;
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{block_first_r,
                                                                                              block_first_theta,
                                                                                              block_first_phi},
                                                                                             {block_first_r + Operator::kBlockExtentR,
                                                                                              block_first_theta + Operator::kBlockExtentTheta,
                                                                                              block_first_phi + Operator::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeReconstruction",
                                 a_policy, the_kernel);
        }

        template <typename View6D>
        static inline void
        DoApply(const Operator& an_operator,
                View6D&&        f5d_s) {
            // TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
            // we have to do something similar.

            // TODO(Etienne M): we can drop the ir_start/ir_end and just pass the range as
            // the arrays are not subarrays.

            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& mask_buffer_r = an_operator.mask_buffer_r_;
            const auto& mask_LIM      = an_operator.mask_LIM_;

            const bool& LIM = an_operator.LIM_;

            const auto the_alpha_solver = an_operator.the_alpha_solver_;

            KOLIOP_ASSERT(the_r_extent % Operator::kBlockExtentR == 0);
            KOLIOP_ASSERT(the_theta_extent % Operator::kBlockExtentTheta == 0);
            KOLIOP_ASSERT(the_phi_extent % Operator::kBlockExtentPhi == 0);

            // TODO(Etienne M): Many opportunities to use constant extent say:
            // ComputeFloatType *[Operator::kFlattenedBlockExtent].

            koliop::utility::View::MDD2<bool> mask_coll{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll"),
                                                        Operator::kBlockExtentR, Operator::kBlockExtentTheta};

            // TODO(Etienne M): Remove the *1D* n, T U Nu etc.
            // Then build a matrix builder which samples the matrix returning
            // the value which we can then cache and use in
            // the_differential_operator and the_rhs. The builder produces the
            // dt/4*T piece of matrix.
            // mask_coll should also be obtain on the fly.

            koliop::utility::View::MDD2<ComputeFloatType> n1d_s{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n1d_s"),
                                                                static_cast<std::size_t>(Operator::kFlattenedBlockExtent), static_cast<std::size_t>(the_species_extent)};
            koliop::utility::View::MDD2<ComputeFloatType> T1d_s{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T1d_s"),
                                                                static_cast<std::size_t>(Operator::kFlattenedBlockExtent), static_cast<std::size_t>(the_species_extent)};
            koliop::utility::View::MDD1<ComputeFloatType> U_par1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par1d"),
                                                                  static_cast<std::size_t>(Operator::kFlattenedBlockExtent)};
            koliop::utility::View::MDD1<ComputeFloatType> q_par_on_nT1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT1d"),
                                                                        static_cast<std::size_t>(Operator::kFlattenedBlockExtent)};
            koliop::utility::View::MDD1<ComputeFloatType> nu1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu1d"),
                                                               static_cast<std::size_t>(Operator::kFlattenedBlockExtent)};

            koliop::utility::View::MDD4<ComputeFloatType> N0{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N0"),
                                                             static_cast<std::size_t>(the_vpar_extent), static_cast<std::size_t>(Operator::kFlattenedBlockExtent), static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree), static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree)};
            koliop::utility::View::MDD4<ComputeFloatType> N1{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                                                             static_cast<std::size_t>(the_vpar_extent), static_cast<std::size_t>(Operator::kFlattenedBlockExtent), static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree), static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree)};
            koliop::utility::View::MDD4<ComputeFloatType> N2{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                                                             static_cast<std::size_t>(the_vpar_extent), static_cast<std::size_t>(Operator::kFlattenedBlockExtent), static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree), static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree)};

            Operator::AlphaSolverType::ScalarMatrixType the_differential_operator{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                                     "the_differential_operator"),
                                                                                  static_cast<std::size_t>(Operator::kBatchCount), static_cast<std::size_t>(the_vpar_extent)};

            Operator::AlphaSolverType::ScalarVectorType the_rhs{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                   "the_rhs"),
                                                                static_cast<std::size_t>(Operator::kBatchCount), static_cast<std::size_t>(the_vpar_extent)};

            // TODO(Etienne M): On GPU, should we just scale the block size ?
            // TODO(Etienne M): We use mask to avoid some computation, but
            // wouldn't it be netter to compute for everything and only in the
            // end, reconstruct f based on the mask ?

            for(OrdinalType block_first_phi = 0;
                block_first_phi < the_phi_extent;
                block_first_phi += Operator::kBlockExtentPhi) {
                for(OrdinalType block_first_theta = 0;
                    block_first_theta < the_theta_extent;
                    block_first_theta += Operator::kBlockExtentTheta) {
                    for(OrdinalType block_first_r = 0;
                        block_first_r < the_r_extent;
                        block_first_r += Operator::kBlockExtentR) {

                        const OrdinalType block_last_r     = block_first_r + Operator::kBlockExtentR;
                        const OrdinalType block_last_theta = block_first_theta + Operator::kBlockExtentTheta;

                        for(OrdinalType itheta = block_first_theta; itheta < block_last_theta; ++itheta) {
                            for(OrdinalType ir = block_first_r; ir < block_last_r; ++ir) {
                                mask_coll(ir - block_first_r, itheta - block_first_theta) = mask_buffer_r(ir) < 0.99;
                            }
                        }

                        if(LIM) {
                            for(OrdinalType itheta = block_first_theta; itheta < block_last_theta; ++itheta) {
                                for(OrdinalType ir = block_first_r; ir < block_last_r; ++ir) {
                                    mask_coll(ir - block_first_r, itheta - block_first_theta) |= mask_LIM(ir, itheta) < 0.99;
                                }
                            }
                        }

                        // NOTE: Skipping would create imbalance
                        // if(all boolean in mask_coll are false) {
                        //     continue;
                        // }

                        {
                            // IN: Constants upon entering the function.
                            // OUT: the_rhs
                            DoComputeAlphaProjection(an_operator,
                                                     block_first_r,
                                                     block_first_theta,
                                                     block_first_phi,
                                                     mask_coll, f5d_s, the_rhs);

                            // IN: Constants upon entering the function.
                            // OUT: n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d
                            DoComputeBlockTo1D(an_operator,
                                               block_first_r,
                                               block_first_theta,
                                               block_first_phi,
                                               mask_coll, n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d);

                            {
                                // call libcoll_N_a_jl(Nspace, vparg, C_jl_, n1d_s, T1d_s, nu1d, N0, N1, N2);

                                // Fence for N0, N1, N2

                                {
                                    // call libcoll_tridiag_matrix(Nspace, dvpar, dt / 4._F64, D_mat, DL_mat, DU1_mat, N0, N1, N2);

                                    the_alpha_solver.Factorize(the_differential_operator.data(), Operator::kFlattenedBlockExtent, the_vpar_extent);
                                }
                                {
                                    // call libcoll_CvCd_RHS(Nspace, dvpar, dt / 4._F64, RHS_vec, N0, N1, N2);
                                }

                                // Fence for RHS_vec and the factorized matrix

                                {
                                    the_alpha_solver.SolveVector(the_differential_operator.data(), Operator::kFlattenedBlockExtent, the_vpar_extent,
                                                                 the_rhs.data());
                                    DoComputeCvCdSource(an_operator,
                                                        T1d_s, U_par1d, q_par_on_nT1d,
                                                        N0, N1, N2,
                                                        the_rhs);
                                    // call libcoll_CvCd_RHS(Nspace, dvpar, dt / 4._F64, RHS_vec, N0, N1, N2);
                                    the_alpha_solver.SolveVector(the_differential_operator.data(), Operator::kFlattenedBlockExtent, the_vpar_extent,
                                                                 the_rhs.data());
                                }
                            }

                            DoComputeReconstruction(an_operator,
                                                    block_first_r,
                                                    block_first_theta,
                                                    block_first_phi,
                                                    mask_coll, f5d_s, the_rhs);

                            // TODO(Etienne M): Fence for DoComputeReconstruction.
                        }
                    }
                }
            }
        }

        void Operator::Apply(Flo64* f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoApply(*this, a_device_visible_f5d_s);

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
