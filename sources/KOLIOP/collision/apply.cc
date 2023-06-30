// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/matrixsampler.h"
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
                        return 1.0 - x;
                    case 2:
                        return 1.0 + ((-2.0 + (1.0 / 2.0) * x) * x);
                    // case 3:
                    //     // Horner -> dependency            : 1 + (-3 + (1.5 - 1/6 * x) * x) * x
                    //     // but FMA friendly.
                    //     // Naive  -> latency and dependency: (1.0 / 6.0) * (-(x * x * x) + 9.0 * x * x - 18.0 * x + 6.0);
                    //     // Slow.
                    //     // Factorized -> low latency       : ((-1.0 / 6.0) * (x - 0.41577455678347908331)) * ((x - 2.29428036027904171982) * (x - 6.28994508293747919686))
                    //     return 1.0 + (-3.0 + ((3.0 / 2.0) - (1.0 / 6.0) * x) * x) * x;
                    default:
                        return -1.0; // TODO(Etienne M): Should we specify some
                                     // garbage?
                }
            }

        } // namespace

        template <typename View2D>
        static inline void
        DoPrepareCollisionMask(const Operator& an_operator,
                               OrdinalType     the_block_first_r,
                               OrdinalType     the_block_first_theta,
                               View2D&&        mask_coll) {

            const auto& LIM = an_operator.LIM_;

            const auto& mask_buffer_r = an_operator.mask_buffer_r_;
            const auto& mask_LIM      = an_operator.mask_LIM_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta) {
                const bool the_mask_value = static_cast<OrdinalType>(mask_buffer_r(ir) < 0.99) |
                                            // NOTE: mask_lim may not be allocated ! We need the && first.
                                            static_cast<OrdinalType>(LIM && (mask_LIM(ir, itheta) < 0.99));

                mask_coll(ir - the_block_first_r,
                          itheta - the_block_first_theta) = the_mask_value;
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{the_block_first_r,
                                                                                              the_block_first_theta},
                                                                                             {the_block_first_r + Constant::kBlockExtentR,
                                                                                              the_block_first_theta + Constant::kBlockExtentTheta}};

            Kokkos::parallel_for("DoPrepareCollisionMask",
                                 a_policy, the_kernel);
        }

        /// Store the current alpha coefficient in the_rhs.
        ///
        template <typename View6D,
                  typename View4D,
                  typename View2D>
        static inline void
        DoComputeAlphaProjection(const Operator& an_operator,
                                 OrdinalType     the_block_first_r,
                                 OrdinalType     the_block_first_theta,
                                 OrdinalType     the_block_first_phi,
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
                const OrdinalType the_block_local_index_r     = ir - the_block_first_r;
                const OrdinalType the_block_local_index_theta = itheta - the_block_first_theta;
                const OrdinalType the_block_local_index_phi   = iphi - the_block_first_phi;

                const OffsetType the_flattened_index = Constant::kBlockExtentR * Constant::kBlockExtentTheta * the_block_local_index_phi +
                                                       Constant::kBlockExtentR * the_block_local_index_theta +
                                                       the_block_local_index_r;

                const OrdinalType the_batch_index         = the_flattened_index / Constant::BatchTraitsType::kBatchSize;
                const OrdinalType the_batch_element_index = the_flattened_index % Constant::BatchTraitsType::kBatchSize;

                if(mask_coll(ir - the_block_first_r,
                             itheta - the_block_first_theta)) {
                    const ComputeFloatType BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                    for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                            the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) = 0.0;
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
                        for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                            the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) = 0.0;
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{the_block_first_r,
                                                                                              the_block_first_theta,
                                                                                              the_block_first_phi},
                                                                                             {the_block_first_r + Constant::kBlockExtentR,
                                                                                              the_block_first_theta + Constant::kBlockExtentTheta,
                                                                                              the_block_first_phi + Constant::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeAlphaProjection",
                                 a_policy, the_kernel);
        }

        template <typename View6D0,
                  typename View6D1,
                  typename View1D>
        static inline void
        DoPrepareMatrix(const Operator&                       an_operator,
                        const MatrixSampler<View6D0, View1D>& the_matrix_sampler,
                        View6D1&&                             the_output_matrix) {

            // TODO(Etienne M): We could reduce the loads by preparing
            // the_output_matrix and the_rhs. Indeed the sampling of the T
            // matrix is the same.

            const auto& the_vpar_extent = an_operator.the_vpar_extent_;

            const auto the_kernel = KOKKOS_LAMBDA(auto the_batch_element_index, auto jpol, auto ipol, auto ivpar, auto the_batch_index) {
                the_matrix_sampler.AllDiagonals(the_batch_index,
                                                ivpar,
                                                ipol, jpol,
                                                the_batch_element_index,
                                                the_output_matrix(the_batch_index,
                                                                  ivpar, 0,
                                                                  ipol, jpol,
                                                                  the_batch_element_index),
                                                the_output_matrix(the_batch_index,
                                                                  ivpar, 1,
                                                                  ipol, jpol,
                                                                  the_batch_element_index),
                                                the_output_matrix(the_batch_index,
                                                                  ivpar, 2,
                                                                  ipol, jpol,
                                                                  the_batch_element_index));
            };

            // TODO(Etienne M): Should I launch 3 kernels, one per diagonal ?
            // This way we could remove the conditional and avoid doing more
            // computation than needed for the first lower and last upper
            // diagonal blocks.

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<5,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0, 0},
                                                                                             {Constant::BatchTraitsType::kBatchSize,
                                                                                              Constant::kMaximumLaguerrePolynomialDegree,
                                                                                              Constant::kMaximumLaguerrePolynomialDegree,
                                                                                              the_vpar_extent,
                                                                                              Constant::kBatchCount}};

            Kokkos::parallel_for("DoPrepareMatrix",
                                 a_policy, the_kernel);
        }

        template <typename View6D,
                  typename View4D>
        static inline void
        DoComputeCvCdSource(const Operator&  an_operator,
                            ComputeFloatType dt,
                            OrdinalType      the_block_first_r,
                            OrdinalType      the_block_first_theta,
                            OrdinalType      the_block_first_phi,
                            const View6D&    the_Najl,
                            View4D&&         the_rhs) {

            const auto& the_vpar_extent = an_operator.the_vpar_extent_;
            const auto& U_par_s         = an_operator.U_par_s_;
            const auto& T_s             = an_operator.T_s_;
            const auto& q_par_on_nT_s   = an_operator.q_par_on_nT_s_;

            const auto& vparg = an_operator.vparg_;

            const auto& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ipol, auto ivpar, auto ir, auto itheta, auto iphi) {

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

                const ComputeFloatType vpar_tmp       = vparg(ivpar);
                const ComputeFloatType Ta             = T_s(ir, itheta, iphi, pSpecies_id);
                const ComputeFloatType U_para         = U_par_s(ir, itheta, iphi, pSpecies_id);
                const ComputeFloatType q_para_on_naTa = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);

                // TODO(Etienne M): Can we simplify that ?

                // TODO(Etienne M): The Najl's matrix indexing is transposed.
                // Careful with N(k, j, l) which is how it is stored and
                // N(k, l, j) how it is used when building the matrix T.

                the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) -= dt / (5.0 * Ta) *
                                                                                  ((5.0 * U_para - 3.0 * q_para_on_naTa) * (vpar_tmp * the_Najl(the_batch_index, ivpar, 0, 0, ipol, the_batch_element_index) + the_Najl(the_batch_index, ivpar, 1, 0, ipol, the_batch_element_index)) +
                                                                                   q_para_on_naTa * vpar_tmp * vpar_tmp / Ta * (vpar_tmp * the_Najl(the_batch_index, ivpar, 0, 0, ipol, the_batch_element_index) + 3.0 * the_Najl(the_batch_index, ivpar, 1, 0, ipol, the_batch_element_index)) +
                                                                                   2.0 * q_para_on_naTa * (3.0 * vpar_tmp / Ta * the_Najl(the_batch_index, ivpar, 2, 0, ipol, the_batch_element_index) - vpar_tmp * the_Najl(the_batch_index, ivpar, 0, 1, ipol, the_batch_element_index) - the_Najl(the_batch_index, ivpar, 1, 1, ipol, the_batch_element_index)));
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<5,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0,
                                                                                              0,
                                                                                              the_block_first_r,
                                                                                              the_block_first_theta,
                                                                                              the_block_first_phi},
                                                                                             {Constant::kMaximumLaguerrePolynomialDegree,
                                                                                              the_vpar_extent,
                                                                                              the_block_first_r + Constant::kBlockExtentR,
                                                                                              the_block_first_theta + Constant::kBlockExtentTheta,
                                                                                              the_block_first_phi + Constant::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeCvCdSource",
                                 a_policy, the_kernel);
        }

        template <typename View6D,
                  typename View4D,
                  typename View2D>
        static inline void
        DoComputeReconstruction(const Operator& an_operator,
                                OrdinalType     the_block_first_r,
                                OrdinalType     the_block_first_theta,
                                OrdinalType     the_block_first_phi,
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

                if(!mask_coll(ir - the_block_first_r,
                              itheta - the_block_first_theta)) {
                    return;
                }

                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // parallelize in vpar too.
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

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

                const ComputeFloatType BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                        const ComputeFloatType muBoT       = mug(imu) * BoT;
                        ComputeFloatType       f5d_s_value = 0.0;

                        for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                            f5d_s_value += the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) *
                                           DoComputeLaguerrePol(ipol, muBoT);
                        }

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = f5d_s_value;
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{the_block_first_r,
                                                                                              the_block_first_theta,
                                                                                              the_block_first_phi},
                                                                                             {the_block_first_r + Constant::kBlockExtentR,
                                                                                              the_block_first_theta + Constant::kBlockExtentTheta,
                                                                                              the_block_first_phi + Constant::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeReconstruction",
                                 a_policy, the_kernel);
        }

        template <typename View6D>
        static inline void
        DoApply(const Operator&  an_operator,
                ComputeFloatType dt,
                View6D&&         f5d_s) {
            // TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
            // we have to do something similar.

            // TODO(Etienne M): we can drop the ir_start/ir_end and just pass the range as
            // the arrays are not subarrays.

            const auto& the_vpar_extent  = an_operator.the_vpar_extent_;
            const auto& the_r_extent     = an_operator.the_r_extent_;
            const auto& the_theta_extent = an_operator.the_theta_extent_;
            const auto& the_phi_extent   = an_operator.the_phi_extent_;

            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;

            const auto& n_s   = an_operator.n_s_;
            const auto& T_s   = an_operator.T_s_;
            const auto& nu_ab = an_operator.nu_ab_;

            const auto the_najl_provider = an_operator.the_najl_provider_;
            const auto the_factorizer    = an_operator.the_factorizer_;
            const auto the_solver        = an_operator.the_solver_;
            const auto the_mv            = an_operator.the_mv_;

            const auto the_stream_pool = an_operator.the_stream_pool_;

            // NOTE: If you have an issue with these assertions, checkout the
            // constant.h file and tune Constant::kBlockExtent[R|Theta|Phi].
            KOLIOP_ASSERT(the_r_extent % Constant::kBlockExtentR == 0);
            KOLIOP_ASSERT(the_theta_extent % Constant::kBlockExtentTheta == 0);
            KOLIOP_ASSERT(the_phi_extent % Constant::kBlockExtentPhi == 0);

            // TODO(Etienne M): Many opportunities to use constant extent say:
            // ComputeFloatType *[Constant::kFlattenedBlockExtent].

            // TODO(Etienne M): couldn't we remove mask_coll. Do redundant
            // computation at the benefit of leaner algorithms ?
            koliop::utility::View::MDD2<bool> mask_coll{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll"),
                                                        Constant::kBlockExtentR, Constant::kBlockExtentTheta};

            // TODO(Etienne M): Storing the Najl cost as much as the matrix ! We
            // should try to compute them on the fly so we can have larger block
            // size.

            // TODO(Etienne M): Many opportunities to use constant extent say:
            // ComputeFloatType *[Constant::kFlattenedBlockExtent].
            typename Constant::MatrixKindType::ScalarType the_Najl{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                      "the_Najl"),
                                                                   static_cast<std::size_t>(Constant::kBatchCount), static_cast<std::size_t>(the_vpar_extent)};

            typename Constant::MatrixKindType::ScalarType the_matrix{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                        "the_matrix"),
                                                                     static_cast<std::size_t>(Constant::kBatchCount), static_cast<std::size_t>(the_vpar_extent)};

            typename Constant::VectorKindType::ScalarType the_alpha_coefficients{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                                    "the_alpha_coefficients"),
                                                                                 static_cast<std::size_t>(Constant::kBatchCount), static_cast<std::size_t>(the_vpar_extent)};

            typename Constant::VectorKindType::ScalarType the_rhs{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                     "the_rhs"),
                                                                  static_cast<std::size_t>(Constant::kBatchCount), static_cast<std::size_t>(the_vpar_extent)};

            // // NOTE: For debug purposes
            // koliop::blas::detail::DoResidualTest(the_factorizer, the_solver, the_mv,
            //                                      the_matrix,
            //                                      the_alpha_coefficients, the_rhs);

            // Samples I - T*dt/4
            const MatrixSampler the_LHS_matrix_sampler{the_Najl,
                                                       coeff_intdvpar,
                                                       -dt / 4.0};

            // Samples I + T*dt/4
            const MatrixSampler the_RHS_matrix_sampler{the_Najl,
                                                       coeff_intdvpar,
                                                       dt / 4.0};

            // TODO(Etienne M): On GPU, should we just scale the block size ?
            // TODO(Etienne M): We use mask to avoid some computation, but
            // wouldn't it be better to compute for everything and only in the
            // end, reconstruct f based on the mask ?
            // TODO(Etienne M): On GPU we want big blocks and a single thread,
            // unparallelized host loop is enough to saturate the GPU. Would
            // this also work correctly on CPU ?

            for(OrdinalType the_block_first_phi = 0;
                the_block_first_phi < the_phi_extent;
                the_block_first_phi += Constant::kBlockExtentPhi) {
                for(OrdinalType the_block_first_theta = 0;
                    the_block_first_theta < the_theta_extent;
                    the_block_first_theta += Constant::kBlockExtentTheta) {
                    for(OrdinalType the_block_first_r = 0;
                        the_block_first_r < the_r_extent;
                        the_block_first_r += Constant::kBlockExtentR) {

                        // TODO(Etienne M): If mask_coll depends on r and theta,
                        // we could bring it up two loops.
                        DoPrepareCollisionMask(an_operator,
                                               the_block_first_r, the_block_first_theta,
                                               mask_coll);

                        // NOTE: Skipping would create imbalance and because we
                        // use large blocks, assuming the resolution is not too
                        // large, we are pretty much assured that we get at
                        // least one point we need to process.
                        // if(all boolean in mask_coll are false) {
                        //     continue;
                        // }

                        {
                            DoComputeAlphaProjection(an_operator,
                                                     the_block_first_r, the_block_first_theta, the_block_first_phi,
                                                     mask_coll, f5d_s, the_alpha_coefficients);

                            {
                                the_najl_provider.Compute(the_stream_pool[0], // TODO(Etienne M): stream tuning
                                                          the_block_first_r, the_block_first_theta, the_block_first_phi,
                                                          n_s, T_s, nu_ab, the_Najl);

                                // Fence for N0, N1, N2

                                {
                                    DoPrepareMatrix(an_operator,
                                                    the_LHS_matrix_sampler,
                                                    the_matrix);

                                    // L*U = (I - T * dt/4)
                                    the_factorizer.Factorize(the_stream_pool[0], // TODO(Etienne M): stream tuning
                                                             the_matrix.data(),
                                                             Constant::kFlattenedBlockExtent, the_vpar_extent);
                                }
                                {
                                    // the_rhs = (I + T * dt/4) * alpha(n)
                                    the_mv.Apply(the_stream_pool[0], //TODO(Etienne M): stream tuning
                                                 1.0,
                                                 the_RHS_matrix_sampler,
                                                 Constant::kFlattenedBlockExtent, the_vpar_extent,
                                                 the_alpha_coefficients.data(),
                                                 // Discards whatever garbage
                                                 // the_rhs could contain.
                                                 0.0,
                                                 the_rhs.data());
                                }

                                // Fence for RHS_vec and the factorized matrix

                                {
                                    // the_rhs = alpha~
                                    the_solver.Solve(the_stream_pool[0], // TODO(Etienne M): stream tuning
                                                     the_matrix.data(),
                                                     Constant::kFlattenedBlockExtent, the_vpar_extent,
                                                     the_rhs.data());

                                    // the_rhs = alpha~~
                                    DoComputeCvCdSource(an_operator,
                                                        dt,
                                                        the_block_first_r, the_block_first_theta, the_block_first_phi,
                                                        the_Najl, the_rhs);

                                    // NOTE: This swap is safe because the
                                    // kernels capture by copy
                                    std::swap(the_rhs, the_alpha_coefficients);

                                    // the_rhs = (I + T * dt/4) * alpha~~
                                    the_mv.Apply(the_stream_pool[0], // TODO(Etienne M): stream tuning
                                                 1.0,
                                                 the_RHS_matrix_sampler,
                                                 Constant::kFlattenedBlockExtent, the_vpar_extent,
                                                 the_alpha_coefficients.data(),
                                                 0.0,
                                                 the_rhs.data());

                                    // the_rhs = alpha(n+1)
                                    the_solver.Solve(the_stream_pool[0], // TODO(Etienne M): stream tuning
                                                     the_matrix.data(),
                                                     Constant::kFlattenedBlockExtent, the_vpar_extent,
                                                     the_rhs.data());
                                }
                            }

                            DoComputeReconstruction(an_operator,
                                                    the_block_first_r, the_block_first_theta, the_block_first_phi,
                                                    mask_coll, f5d_s, the_rhs);

                            // TODO(Etienne M): Fence for DoComputeReconstruction.
                            Kokkos::fence();
                        }
                    }
                }
            }
        }

        void Operator::Apply(ComputeFloatType dt,
                             Flo64*           f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoApply(*this, dt, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s,
                                           koliop::utility::View::UHL6<Flo64>{f5d_s,
                                                                              static_cast<std::size_t>(the_mu_extent_), static_cast<std::size_t>(the_vpar_extent_),
                                                                              static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                              static_cast<std::size_t>(the_species_extent_)});

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
