// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_CVCDDRIVER_H
#define KOLIOP_COLLISION_CVCDDRIVER_H

#include "KOLIOP/blas/mv.h"
#include "KOLIOP/blas/trf.h"
#include "KOLIOP/blas/trsv.h"
#include "KOLIOP/collision/matrixsampler.h"
#include "KOLIOP/collision/najl.h"
#include "KOLIOP/utility/select.h"

namespace koliop {
    namespace collision {
        namespace detail {
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

            template <typename CvCdDriver,
                      typename Operator,
                      typename ExecutionSpace,
                      typename View2D>
            static inline void
            DoPrepareCollisionMask(const Operator& an_operator,
                                   ExecutionSpace  an_execution_space,
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
                                                                         Kokkos::Iterate::Default,
                                                                         Kokkos::Iterate::Default>>{an_execution_space,
                                                                                                    {the_block_first_r,
                                                                                                     the_block_first_theta},
                                                                                                    {the_block_first_r + CvCdDriver::kBlockExtentR,
                                                                                                     the_block_first_theta + CvCdDriver::kBlockExtentTheta}};

                Kokkos::parallel_for("DoPrepareCollisionMask",
                                     a_policy, the_kernel);
            }

            /// Store the current alpha coefficient in the_rhs.
            ///
            template <typename CvCdDriver,
                      typename Operator,
                      typename ExecutionSpace,
                      typename View6D,
                      typename View4D,
                      typename View2D>
            static inline void
            DoComputeAlphaProjection(const Operator& an_operator,
                                     ExecutionSpace  an_execution_space,
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

                // TODO(Etienne M): Make sure the accesses are coalesced on GPU
                // and sequential on CPU.
                // Using an MDRange policy and taking into account the
                // the_batch_element_index, should lead to very decent accesses.
                // TODO(Etienne M): Maybe we could get better performance by
                // replacing the MDRange policy with a TeamPolicy and our own
                // indices computation.

                const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType the_block_local_index_r     = ir - the_block_first_r;
                    const OrdinalType the_block_local_index_theta = itheta - the_block_first_theta;
                    const OrdinalType the_block_local_index_phi   = iphi - the_block_first_phi;

                    const OffsetType the_flattened_index = CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta * the_block_local_index_phi +
                                                           CvCdDriver::kBlockExtentR * the_block_local_index_theta +
                                                           the_block_local_index_r;

                    const OrdinalType the_batch_element_index = (the_flattened_index) % CvCdDriver::BatchTraitsType::kBatchSize;
                    const OrdinalType the_batch_index         = (the_flattened_index / (CvCdDriver::BatchTraitsType::kBatchSize)) /* % CvCdDriver::kBatchCount */;

                    if(mask_coll(ir - the_block_first_r,
                                 itheta - the_block_first_theta)) {
                        const ComputeFloatType BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                        for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                                the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) = 0.0;
                                for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                                    const ComputeFloatType muBoT = mug(imu) * BoT;

                                    // TODO(Etienne M): Expensive
                                    // DoComputeLaguerrePol call.
                                    // NOTE: This is can be mitigated with
                                    // unrolling. Also, on GPUs, we should not get
                                    // mich thread divergence except if mask_coll
                                    // has a *wild* pattern.
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
                                                                         Kokkos::Iterate::Default,
                                                                         Kokkos::Iterate::Default>>{an_execution_space,
                                                                                                    {the_block_first_r,
                                                                                                     the_block_first_theta,
                                                                                                     the_block_first_phi},
                                                                                                    {the_block_first_r + CvCdDriver::kBlockExtentR,
                                                                                                     the_block_first_theta + CvCdDriver::kBlockExtentTheta,
                                                                                                     the_block_first_phi + CvCdDriver::kBlockExtentPhi}};

                Kokkos::parallel_for("DoComputeAlphaProjection",
                                     a_policy, the_kernel);
            }

            template <typename CvCdDriver,
                      typename Operator,
                      typename ExecutionSpace,
                      typename View6D0,
                      typename View6D1,
                      typename View1D>
            static inline void
            DoPrepareMatrix(const Operator&                       an_operator,
                            ExecutionSpace                        an_execution_space,
                            const MatrixSampler<View6D0, View1D>& the_matrix_sampler,
                            View6D1&&                             the_output_matrix) {

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

                const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<5,
                                                                         Kokkos::Iterate::Default,
                                                                         Kokkos::Iterate::Default>>{an_execution_space,
                                                                                                    {0, 0, 0, 0, 0},
                                                                                                    {CvCdDriver::BatchTraitsType::kBatchSize,
                                                                                                     Constant::kMaximumLaguerrePolynomialDegree,
                                                                                                     Constant::kMaximumLaguerrePolynomialDegree,
                                                                                                     the_vpar_extent,
                                                                                                     CvCdDriver::kBatchCount}};

                Kokkos::parallel_for("DoPrepareMatrix",
                                     a_policy, the_kernel);
            }

            template <typename CvCdDriver,
                      typename Operator,
                      typename ExecutionSpace,
                      typename View6D,
                      typename View4D>
            static inline void
            DoComputeCvCdSource(const Operator&  an_operator,
                                ExecutionSpace   an_execution_space,
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
                    // the modulo with faster instructions.
                    const OrdinalType the_block_local_index_r     = ir - the_block_first_r;
                    const OrdinalType the_block_local_index_theta = itheta - the_block_first_theta;
                    const OrdinalType the_block_local_index_phi   = iphi - the_block_first_phi;

                    const OffsetType the_flattened_index = CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta * the_block_local_index_phi +
                                                           CvCdDriver::kBlockExtentR * the_block_local_index_theta +
                                                           the_block_local_index_r;

                    const OrdinalType the_batch_element_index = (the_flattened_index) % CvCdDriver::BatchTraitsType::kBatchSize;
                    const OrdinalType the_batch_index         = (the_flattened_index / (CvCdDriver::BatchTraitsType::kBatchSize)) /* % CvCdDriver::kBatchCount */;

                    const ComputeFloatType vpar_tmp       = vparg(ivpar);
                    const ComputeFloatType Ta             = T_s(ir, itheta, iphi, pSpecies_id);
                    const ComputeFloatType U_para         = U_par_s(ir, itheta, iphi, pSpecies_id);
                    const ComputeFloatType q_para_on_naTa = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);

                    // TODO(Etienne M): Can we simplify that ?

                    the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) -= dt / (5.0 * Ta) *
                                                                                      ((5.0 * U_para - 3.0 * q_para_on_naTa) * (vpar_tmp * the_Najl(the_batch_index, ivpar, 0, ipol, 0, the_batch_element_index) + the_Najl(the_batch_index, ivpar, 1, ipol, 0, the_batch_element_index)) +
                                                                                       q_para_on_naTa * vpar_tmp * vpar_tmp / Ta * (vpar_tmp * the_Najl(the_batch_index, ivpar, 0, ipol, 0, the_batch_element_index) + 3.0 * the_Najl(the_batch_index, ivpar, 1, ipol, 0, the_batch_element_index)) +
                                                                                       2.0 * q_para_on_naTa * (3.0 * vpar_tmp / Ta * the_Najl(the_batch_index, ivpar, 2, ipol, 0, the_batch_element_index) - vpar_tmp * the_Najl(the_batch_index, ivpar, 0, ipol, 1, the_batch_element_index) - the_Najl(the_batch_index, ivpar, 1, ipol, 1, the_batch_element_index)));
                };

                const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<5,
                                                                         Kokkos::Iterate::Default,
                                                                         Kokkos::Iterate::Default>>{an_execution_space,
                                                                                                    {0,
                                                                                                     0,
                                                                                                     the_block_first_r,
                                                                                                     the_block_first_theta,
                                                                                                     the_block_first_phi},
                                                                                                    {Constant::kMaximumLaguerrePolynomialDegree,
                                                                                                     the_vpar_extent,
                                                                                                     the_block_first_r + CvCdDriver::kBlockExtentR,
                                                                                                     the_block_first_theta + CvCdDriver::kBlockExtentTheta,
                                                                                                     the_block_first_phi + CvCdDriver::kBlockExtentPhi}};

                Kokkos::parallel_for("DoComputeCvCdSource",
                                     a_policy, the_kernel);
            }

            template <typename CvCdDriver,
                      typename Operator,
                      typename ExecutionSpace,
                      typename View6D,
                      typename View4D,
                      typename View2D>
            static inline void
            DoComputeReconstruction(const Operator& an_operator,
                                    ExecutionSpace  an_execution_space,
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

                // TODO(Etienne M): Make sure the accesses are coalesced on GPU
                // and sequential on CPU.

                const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {

                    const OrdinalType the_team_index = a_team_handle.league_rank();

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType the_block_local_index_ir     = (the_team_index) % CvCdDriver::kBlockExtentR;
                    const OrdinalType the_block_local_index_itheta = (the_team_index / (CvCdDriver::kBlockExtentR)) % CvCdDriver::kBlockExtentTheta;
                    const OrdinalType the_block_local_index_iphi   = (the_team_index / (CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta)) /* % CvCdDriver::kBlockExtentPhi */;

                    if(!mask_coll(the_block_local_index_ir,
                                  the_block_local_index_itheta)) {
                        // NOTE: this return does not break the rest of the function
                        // as all the thread of the team will execute the same
                        // branch.
                        return;
                    }

                    const OffsetType the_flattened_index = the_team_index;

                    const OrdinalType the_batch_element_index = (the_flattened_index) % CvCdDriver::BatchTraitsType::kBatchSize;
                    const OrdinalType the_batch_index         = (the_flattened_index / (CvCdDriver::BatchTraitsType::kBatchSize)) /* % CvCdDriver::kBatchCount */;

                    const OrdinalType ir     = the_block_first_r + the_block_local_index_ir;
                    const OrdinalType itheta = the_block_first_theta + the_block_local_index_itheta;
                    const OrdinalType iphi   = the_block_first_phi + the_block_local_index_iphi;

                    const ComputeFloatType BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
                                                  std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                    the_mu_extent, the_vpar_extent),
                        [&](auto imu, auto ivpar) {
                            const ComputeFloatType muBoT       = mug(imu) * BoT;
                            ComputeFloatType       f5d_s_value = 0.0;

                            for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.
                                f5d_s_value += the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) *
                                               DoComputeLaguerrePol(ipol, muBoT);
                            }

                            f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = f5d_s_value;
                        });
                };

                const auto the_team_count = CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta * CvCdDriver::kBlockExtentPhi;

                // Due to the inherent Cuda/HIP thread block limit.
                KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

                // TODO(Etienne M): TEAM hardcoded size, check for auto tuning
                // opportunities.
                const auto a_policy = Kokkos::TeamPolicy<Constant::DeviceExecutionSpaceType>{an_execution_space,
                                                                                             static_cast<int>(the_team_count),
                                                                                             koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, 384)};

                Kokkos::parallel_for("DoComputeReconstruction",
                                     a_policy, the_kernel);
            }

        } // namespace detail


        ////////////////////////////////////////////////////////////////////////
        // CvCdDriver definition
        ////////////////////////////////////////////////////////////////////////

        template <OrdinalType the_block_extent_r,
                  OrdinalType the_block_extent_theta,
                  OrdinalType the_block_extent_phi,
                  typename BatchTraits>
        class CvCdDriver {
        public:
            static inline constexpr OrdinalType kBlockExtentR     = the_block_extent_r;
            static inline constexpr OrdinalType kBlockExtentTheta = the_block_extent_theta;
            static inline constexpr OrdinalType kBlockExtentPhi   = the_block_extent_phi;

            static inline constexpr OffsetType kFlattenedBlockExtent = kBlockExtentR * kBlockExtentTheta * kBlockExtentPhi;

            using BatchTraitsType = BatchTraits;

            static inline constexpr OffsetType kBatchCount = BatchTraitsType::BatchCount(kFlattenedBlockExtent);
            static_assert(kBatchCount * BatchTraitsType::kBatchSize == kFlattenedBlockExtent, "");

            using VectorKindType = koliop::blas::kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                                   Constant::kMaximumLaguerrePolynomialDegree>;

            using MatrixKindType = koliop::blas::kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                              Constant::kMaximumLaguerrePolynomialDegree>;

        protected:
            using MatrixSampler = MatrixSampler<typename MatrixKindType::ScalarType,
                                                koliop::utility::View::InternalInputL1<const Flo64>>;

        public:
            CvCdDriver(OrdinalType                                          pSpecies_id,
                       OrdinalType                                          the_vpar_extent,
                       OrdinalType                                          the_species_extent,
                       bool                                                 collision_interspecies,
                       const Flo64*                                         comb_mat,
                       koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                       koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                       koliop::utility::View::InternalInputL1<const Flo64>& vparg,
                       koliop::utility::View::InternalInputL1<const Flo64>& coeff_intdvpar);

            template <typename Operator, // Templatized to avoid dependency loops
                      typename ExecutionSpace,
                      typename View6D>
            void Apply(const Operator&  an_operator,
                       ExecutionSpace   an_execution_space,
                       OrdinalType      the_block_first_r,
                       OrdinalType      the_block_first_theta,
                       OrdinalType      the_block_first_phi,
                       ComputeFloatType dt,
                       View6D&&         f5d_s);

        protected:
            // Miscellaneous numerical tooling

            Najl<CvCdDriver> the_najl_provider_;

            koliop::utility::View::MDD<bool[kBlockExtentR][kBlockExtentTheta]> mask_coll_;

            // NOTE: We know of an additional constant extent, the batch count
            // but we cannot exploit it due to kokkos enforcing that constant
            // extent must be placed after dynamic extent. vpar is non constant
            // and is placed after the batch count.
            typename MatrixKindType::ScalarType the_Najl_;
            typename MatrixKindType::ScalarType the_matrix_;
            typename VectorKindType::ScalarType the_alpha_coefficients_;
            typename VectorKindType::ScalarType the_rhs_;

            koliop::blas::BatchedBlockTridiagonalByBlockTRF<BatchTraitsType, Constant::kMaximumLaguerrePolynomialDegree>              the_factorizer_;
            koliop::blas::BatchedBlockTridiagonalByBlockTRSV<BatchTraitsType, Constant::kMaximumLaguerrePolynomialDegree>             the_solver_;
            koliop::blas::BatchedMatrixSamplingBlockTridiagonalByBlockMV<BatchTraitsType, Constant::kMaximumLaguerrePolynomialDegree> the_mv_;

            MatrixSampler the_LHS_matrix_sampler_;
            MatrixSampler the_RHS_matrix_sampler_;
        };


        ////////////////////////////////////////////////////////////////////////
        // CvCdDriver methods definition
        ////////////////////////////////////////////////////////////////////////

        template <OrdinalType the_block_extent_r,
                  OrdinalType the_block_extent_theta,
                  OrdinalType the_block_extent_phi,
                  typename BatchTraits>
        CvCdDriver<the_block_extent_r,
                   the_block_extent_theta,
                   the_block_extent_phi,
                   BatchTraits>::CvCdDriver(OrdinalType                                          pSpecies_id,
                                            OrdinalType                                          the_vpar_extent,
                                            OrdinalType                                          the_species_extent,
                                            bool                                                 collision_interspecies,
                                            const Flo64*                                         comb_mat,
                                            koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                                            koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                                            koliop::utility::View::InternalInputL1<const Flo64>& vparg,
                                            koliop::utility::View::InternalInputL1<const Flo64>& coeff_intdvpar)
            : the_najl_provider_{pSpecies_id,
                                 the_vpar_extent, the_species_extent,
                                 collision_interspecies,
                                 comb_mat,
                                 hat_As, hat_Zs, vparg}
            , mask_coll_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll")}
            , the_Najl_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                           "the_Najl"),
                        static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(the_vpar_extent)}
            , the_matrix_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "the_matrix"),
                          static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(the_vpar_extent)}
            , the_alpha_coefficients_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                         "the_alpha_coefficients"),
                                      static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(the_vpar_extent)}
            , the_rhs_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                          "the_rhs"),
                       static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(the_vpar_extent)}
            , the_factorizer_{}
            , the_solver_{}
            , the_mv_{}
            , the_LHS_matrix_sampler_{the_Najl_,
                                      coeff_intdvpar}
            , the_RHS_matrix_sampler_{the_Najl_,
                                      coeff_intdvpar} {
            // EMPTY
        }

        template <OrdinalType the_block_extent_r,
                  OrdinalType the_block_extent_theta,
                  OrdinalType the_block_extent_phi,
                  typename BatchTraits>
        template <typename Operator, // Templatized to avoid dependency loops
                  typename ExecutionSpace,
                  typename View6D>
        void CvCdDriver<the_block_extent_r,
                        the_block_extent_theta,
                        the_block_extent_phi,
                        BatchTraits>::Apply(const Operator&  an_operator,
                                            ExecutionSpace   an_execution_space,
                                            OrdinalType      the_block_first_r,
                                            OrdinalType      the_block_first_theta,
                                            OrdinalType      the_block_first_phi,
                                            ComputeFloatType dt,
                                            View6D&&         f5d_s) {

            const auto& the_vpar_extent  = an_operator.the_vpar_extent_;
            const auto& the_r_extent     = an_operator.the_r_extent_;
            const auto& the_theta_extent = an_operator.the_theta_extent_;
            const auto& the_phi_extent   = an_operator.the_phi_extent_;

            const auto& n_s   = an_operator.n_s_;
            const auto& T_s   = an_operator.T_s_;
            const auto& nu_ab = an_operator.nu_ab_;

            // NOTE: If you have an issue with these assertions, checkout the
            // constant.h file and tune kBlockExtent[R|Theta|Phi].
            KOLIOP_ASSERT(the_r_extent % kBlockExtentR == 0);
            KOLIOP_ASSERT(the_theta_extent % kBlockExtentTheta == 0);
            KOLIOP_ASSERT(the_phi_extent % kBlockExtentPhi == 0);

            KOLIOP_ASSERT((the_block_first_r + kBlockExtentR) <= the_r_extent);
            KOLIOP_ASSERT((the_block_first_theta + kBlockExtentTheta) <= the_theta_extent);
            KOLIOP_ASSERT((the_block_first_phi + kBlockExtentPhi) <= the_phi_extent);

            // In the future, the time step could vary.
            the_LHS_matrix_sampler_.DeltaTime() = static_cast<ComputeFloatType>(-dt / 4.0);
            the_RHS_matrix_sampler_.DeltaTime() = static_cast<ComputeFloatType>(dt / 4.0);

            // TODO(Etienne M): If mask_coll depends on r and theta,
            // we could bring it up two loops into Apply.
            detail::DoPrepareCollisionMask<CvCdDriver>(an_operator,
                                                       an_execution_space,
                                                       the_block_first_r, the_block_first_theta,
                                                       mask_coll_);

            // NOTE: Skipping would create imbalance and because we
            // use large blocks, assuming the resolution is not too
            // large, we are pretty much assured that we get at
            // least one point we need to process.
            // if(all boolean in mask_coll are false) {
            //     continue;
            // }

            {
                detail::DoComputeAlphaProjection<CvCdDriver>(an_operator,
                                                             an_execution_space,
                                                             the_block_first_r,
                                                             the_block_first_theta, the_block_first_phi,
                                                             mask_coll_, f5d_s, the_alpha_coefficients_);
                {
                    the_najl_provider_.Compute(an_execution_space,
                                               the_block_first_r, the_block_first_theta, the_block_first_phi,
                                               n_s, T_s, nu_ab, the_Najl_);

                    // Fence for N0, N1, N2

                    {
                        detail::DoPrepareMatrix<CvCdDriver>(an_operator,
                                                            an_execution_space,
                                                            the_LHS_matrix_sampler_,
                                                            the_matrix_);

                        // L*U = (I - T * dt/4)
                        the_factorizer_.Factorize(an_execution_space,
                                                  the_matrix_.data(),
                                                  kFlattenedBlockExtent, the_vpar_extent);
                    }
                    {
                        // the_rhs = (I + T * dt/4) * alpha(n)
                        the_mv_.Apply(an_execution_space,
                                      1.0,
                                      the_RHS_matrix_sampler_,
                                      kFlattenedBlockExtent, the_vpar_extent,
                                      the_alpha_coefficients_.data(),
                                      // Discards whatever garbage
                                      // the_rhs could contain.
                                      0.0,
                                      the_rhs_.data());
                    }

                    // Fence for RHS_vec and the factorized matrix

                    {
                        // the_rhs = alpha~
                        the_solver_.Solve(an_execution_space,
                                          the_matrix_.data(),
                                          kFlattenedBlockExtent, the_vpar_extent,
                                          the_rhs_.data());

                        // the_rhs = alpha~~
                        detail::DoComputeCvCdSource<CvCdDriver>(an_operator,
                                                                an_execution_space,
                                                                dt,
                                                                the_block_first_r, the_block_first_theta, the_block_first_phi,
                                                                the_Najl_, the_rhs_);

                        // NOTE: This swap is safe because the
                        // kernels capture by copy
                        std::swap(the_rhs_, the_alpha_coefficients_);

                        // the_rhs = (I + T * dt/4) * alpha~~
                        the_mv_.Apply(an_execution_space,
                                      1.0,
                                      the_RHS_matrix_sampler_,
                                      kFlattenedBlockExtent, the_vpar_extent,
                                      the_alpha_coefficients_.data(),
                                      0.0,
                                      the_rhs_.data());

                        // the_rhs = alpha(n+1)
                        the_solver_.Solve(an_execution_space,
                                          the_matrix_.data(),
                                          kFlattenedBlockExtent, the_vpar_extent,
                                          the_rhs_.data());
                    }
                }

                detail::DoComputeReconstruction<CvCdDriver>(an_operator,
                                                            an_execution_space,
                                                            the_block_first_r,
                                                            the_block_first_theta, the_block_first_phi,
                                                            mask_coll_, f5d_s, the_rhs_);
            }
        }

    } // namespace collision
} // namespace koliop

#endif
