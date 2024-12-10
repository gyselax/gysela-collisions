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
                    switch(l) {
                        case 0:
                            return 1.0;
                        case 1:
                            return 1.0 - x;
                        case 2:
                            // return 0.5 * (x - 0.5857864376269049511983112758) * (x - 3.414213562373095048801688724);
                            return 1.0 + ((-2.0 + (1.0 / 2.0) * x) * x);
                        // case 3:
                        //     // Horner -> dependency            : 1 + (-3 + (1.5 - 1/6 * x) * x) * x
                        //     // but FMA friendly.
                        //     // Naive  -> latency and dependency: (1.0 / 6.0) * (-(x * x * x) + 9.0 * x * x - 18.0 * x + 6.0);
                        //     // Slow.
                        //     // Factorized -> low latency       : ((-1.0 / 6.0) * (x - 0.41577455678347908331)) * ((x - 2.29428036027904171982) * (x - 6.28994508293747919686))
                        //     return 1.0 + (-3.0 + ((3.0 / 2.0) - (1.0 / 6.0) * x) * x) * x;
                        default:
                            return 0.0;
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
                                   OrdinalType     block_first_r,
                                   OrdinalType     block_first_theta,
                                   View2D&&        mask_coll) {

                const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
                const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

                const auto& LIM = an_operator.LIM_;

                const auto& mask_buffer_r = an_operator.mask_buffer_r_;
                const auto& mask_LIM      = an_operator.mask_LIM_;

                const auto kernel = KOKKOS_LAMBDA(OrdinalType ir, OrdinalType itheta) {
                    const bool mask_value = static_cast<OrdinalType>(mask_buffer_r(local_domain_r_offset + ir) < 0.99) |
                                            // NOTE: mask_lim may not be allocated ! We need the && first.
                                            static_cast<OrdinalType>(LIM && (mask_LIM(local_domain_r_offset + ir,
                                                                                      local_domain_theta_offset + itheta) < 0.99));

                    mask_coll(ir - block_first_r,
                              itheta - block_first_theta) = mask_value;
                };

                const auto a_policy = Kokkos::MDRangePolicy<
                    // Kokkos::LaunchBounds<256, 1>, Kokkos::IndexType<OrdinalType>,
                    ExecutionSpace,
                    Kokkos::Rank<2,
                                 Kokkos::Iterate::Left,
                                 Kokkos::Iterate::Left>>{an_execution_space,
                                                         {block_first_r,
                                                          block_first_theta},
                                                         {block_first_r + CvCdDriver::kBlockExtentR,
                                                          block_first_theta + CvCdDriver::kBlockExtentTheta}};

                Kokkos::parallel_for("DoPrepareCollisionMask",
                                     a_policy, kernel);
            }

            /// Store the current alpha coefficient in rhs.
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
                                     OrdinalType     block_first_r,
                                     OrdinalType     block_first_theta,
                                     OrdinalType     block_first_phi,
                                     OrdinalType     species_index,
                                     const View2D&   mask_coll,
                                     const View6D&   f5d_s,
                                     View4D&&        rhs) {

                const auto& mu_extent   = an_operator.mu_extent_;
                const auto& vpar_extent = an_operator.vpar_extent_;

                const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
                const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

                const auto& mug          = an_operator.mug_;
                const auto& coeff_intdmu = an_operator.coeff_intdmu_;
                const auto& B_norm       = an_operator.B_norm_;
                const auto& T_s          = an_operator.T_s_;

                // TODO(Etienne M): Make sure the accesses are sequential on
                // CPU.

                const auto team_count = CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta * CvCdDriver::kBlockExtentPhi;

                // Due to the inherent Cuda/HIP thread block limit.
                KOLIOP_ASSERT(team_count <= std::numeric_limits<int>::max());

                using PolicyType = Kokkos::TeamPolicy<
                    // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                    ExecutionSpace>;

                const auto a_policy = PolicyType{
                    an_execution_space,
                    static_cast<int>(team_count),
                    // NOTE: The thread block size and the Nmu benefit from being
                    // close. This is due to the TeamThreadMDRange which
                    // parallelizes only the left most index which is Nmu.
                    koliop::utility::Select<ExecutionSpace>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

                const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                    const OrdinalType team_index = a_team_handle.league_rank();

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType block_local_index_ir     = (team_index) % CvCdDriver::kBlockExtentR;
                    const OrdinalType block_local_index_itheta = (team_index / (CvCdDriver::kBlockExtentR)) % CvCdDriver::kBlockExtentTheta;
                    const OrdinalType block_local_index_iphi   = (team_index / (CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta)) /* % CvCdDriver::kBlockExtentPhi */;

                    const OffsetType flattened_index = team_index;

                    const OrdinalType batch_element_index = (flattened_index) % CvCdDriver::BatchTraitsType::kBatchSize;
                    const OrdinalType batch_index         = (flattened_index / (CvCdDriver::BatchTraitsType::kBatchSize)) /* % CvCdDriver::kBatchCount */;

                    const OrdinalType ir     = block_first_r + block_local_index_ir;
                    const OrdinalType itheta = block_first_theta + block_local_index_itheta;
                    const OrdinalType iphi   = block_first_phi + block_local_index_iphi;

                    if(mask_coll(block_local_index_ir,
                                 block_local_index_itheta)) {

                        const ComputeFloatType BoT = B_norm(local_domain_r_offset + ir,
                                                            local_domain_theta_offset + itheta) /
                                                     T_s(ir, itheta, iphi, species_index);

                        for(OrdinalType ivpar = 0; ivpar < vpar_extent; ++ivpar) {
                            for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {

                                ComputeFloatType              rhs_value;
                                Kokkos::Sum<ComputeFloatType> rhs_value_reducer{rhs_value};
                                rhs_value_reducer.init(rhs_value_reducer.reference());

                                Kokkos::parallel_for(
                                    Kokkos::TeamThreadRange(a_team_handle,
                                                            mu_extent),
                                    [&](auto imu) {
                                        const ComputeFloatType muBoT     = mug(imu) * BoT;
                                        const ComputeFloatType coeff_tot = DoComputeLaguerrePol(ipol, muBoT) * BoT * coeff_intdmu(imu) * Kokkos::exp(-muBoT);

                                        rhs_value_reducer.join(rhs_value_reducer.reference(),
                                                               coeff_tot * f5d_s(imu, ivpar, ir, itheta, iphi, species_index));
                                    });

                                a_team_handle.team_reduce(rhs_value_reducer);

                                // NOTE: This should be a no-op.
                                rhs_value = rhs_value_reducer.reference();

                                Kokkos::single(
                                    Kokkos::PerTeam(a_team_handle),
                                    [&]() {
                                        rhs(batch_index, ivpar, ipol, batch_element_index) = rhs_value;
                                    });
                            }
                        }
                    } else {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(a_team_handle,
                                                    vpar_extent),
                            [&](auto ivpar) {
                                for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                                    // NOTE: These accesses are .. bad.
                                    rhs(batch_index, ivpar, ipol, batch_element_index) = 0.0;
                                }
                            });
                    }
                };

                Kokkos::parallel_for("DoComputeAlphaProjection",
                                     a_policy, kernel);
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
                            const MatrixSampler<View6D0, View1D>& matrix_sampler,
                            View6D1&&                             output_matrix) {

                const auto& vpar_extent = an_operator.vpar_extent_;

                const auto kernel = KOKKOS_LAMBDA(OrdinalType batch_element_index, OrdinalType jpol, OrdinalType ipol, OrdinalType ivpar, OrdinalType batch_index) {
                    matrix_sampler.AllDiagonals(batch_index,
                                                ivpar,
                                                ipol, jpol,
                                                batch_element_index,
                                                output_matrix(batch_index,
                                                              ivpar, 0,
                                                              ipol, jpol,
                                                              batch_element_index),
                                                output_matrix(batch_index,
                                                              ivpar, 1,
                                                              ipol, jpol,
                                                              batch_element_index),
                                                output_matrix(batch_index,
                                                              ivpar, 2,
                                                              ipol, jpol,
                                                              batch_element_index));
                };

                const auto a_policy = Kokkos::MDRangePolicy<
                    // Kokkos::LaunchBounds<256, 1>, Kokkos::IndexType<OrdinalType>,
                    ExecutionSpace,
                    Kokkos::Rank<5,
                                 Kokkos::Iterate::Left,
                                 Kokkos::Iterate::Left>>{an_execution_space,
                                                         {0, 0, 0, 0, 0},
                                                         {CvCdDriver::BatchTraitsType::kBatchSize,
                                                          Constant::kMaximumLaguerrePolynomialDegree,
                                                          Constant::kMaximumLaguerrePolynomialDegree,
                                                          vpar_extent,
                                                          CvCdDriver::kBatchCount}};

                Kokkos::parallel_for("DoPrepareMatrix",
                                     a_policy, kernel);
            }

            template <typename CvCdDriver,
                      typename Operator,
                      typename ExecutionSpace,
                      typename View6D,
                      typename View4D>
            static inline void
            DoComputeCvCdSource(const Operator&  an_operator,
                                ExecutionSpace   an_execution_space,
                                OrdinalType      block_first_r,
                                OrdinalType      block_first_theta,
                                OrdinalType      block_first_phi,
                                ComputeFloatType dt,
                                OrdinalType      species_index,
                                const View6D&    Najl,
                                View4D&&         rhs) {

                const auto& vpar_extent   = an_operator.vpar_extent_;
                const auto& U_par_s       = an_operator.U_par_s_;
                const auto& T_s           = an_operator.T_s_;
                const auto& q_par_on_nT_s = an_operator.q_par_on_nT_s_;

                const auto& vparg = an_operator.vparg_;

                const auto kernel = KOKKOS_LAMBDA(OrdinalType ipol, OrdinalType ivpar, OrdinalType ir, OrdinalType itheta, OrdinalType iphi) {

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType block_local_index_ir     = ir - block_first_r;
                    const OrdinalType block_local_index_itheta = itheta - block_first_theta;
                    const OrdinalType block_local_index_iphi   = iphi - block_first_phi;

                    const OffsetType flattened_index = CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta * block_local_index_iphi +
                                                       CvCdDriver::kBlockExtentR * block_local_index_itheta +
                                                       block_local_index_ir;

                    const OrdinalType batch_element_index = (flattened_index) % CvCdDriver::BatchTraitsType::kBatchSize;
                    const OrdinalType batch_index         = (flattened_index / (CvCdDriver::BatchTraitsType::kBatchSize)) /* % CvCdDriver::kBatchCount */;

                    const ComputeFloatType vpar_tmp       = vparg(ivpar);
                    const ComputeFloatType Ta             = T_s(ir, itheta, iphi, species_index);
                    const ComputeFloatType U_para         = U_par_s(ir, itheta, iphi, species_index);
                    const ComputeFloatType q_para_on_naTa = q_par_on_nT_s(ir, itheta, iphi, species_index);

                    // TODO(Etienne M): Can we simplify that ?

                    rhs(batch_index, ivpar, ipol, batch_element_index) -= dt / (5.0 * Ta) *
                                                                          ((5.0 * U_para - 3.0 * q_para_on_naTa) * (vpar_tmp * Najl(batch_index, ivpar, 0, ipol, 0, batch_element_index) + Najl(batch_index, ivpar, 1, ipol, 0, batch_element_index)) +
                                                                           q_para_on_naTa * vpar_tmp * vpar_tmp / Ta * (vpar_tmp * Najl(batch_index, ivpar, 0, ipol, 0, batch_element_index) + 3.0 * Najl(batch_index, ivpar, 1, ipol, 0, batch_element_index)) +
                                                                           2.0 * q_para_on_naTa * (3.0 * vpar_tmp / Ta * Najl(batch_index, ivpar, 2, ipol, 0, batch_element_index) - vpar_tmp * Najl(batch_index, ivpar, 0, ipol, 1, batch_element_index) - Najl(batch_index, ivpar, 1, ipol, 1, batch_element_index)));
                };

                const auto a_policy = Kokkos::MDRangePolicy<
                    // NOTE: Weird stuff happens when I set LaunchBounds <64,1>.
                    // Notably, the kernel is not launched ??
                    // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                    ExecutionSpace,
                    Kokkos::Rank<5,
                                 Kokkos::Iterate::Left,
                                 Kokkos::Iterate::Left>>{an_execution_space,
                                                         {0,
                                                          0,
                                                          block_first_r,
                                                          block_first_theta,
                                                          block_first_phi},
                                                         {Constant::kMaximumLaguerrePolynomialDegree,
                                                          vpar_extent,
                                                          block_first_r + CvCdDriver::kBlockExtentR,
                                                          block_first_theta + CvCdDriver::kBlockExtentTheta,
                                                          block_first_phi + CvCdDriver::kBlockExtentPhi}};

                Kokkos::parallel_for("DoComputeCvCdSource",
                                     a_policy, kernel);
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
                                    OrdinalType     block_first_r,
                                    OrdinalType     block_first_theta,
                                    OrdinalType     block_first_phi,
                                    OrdinalType     species_index,
                                    const View2D&   mask_coll,
                                    const View4D&   rhs,
                                    View6D&&        f5d_s) {

                const auto& mu_extent   = an_operator.mu_extent_;
                const auto& vpar_extent = an_operator.vpar_extent_;

                const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
                const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

                const auto& mug    = an_operator.mug_;
                const auto& B_norm = an_operator.B_norm_;
                const auto& T_s    = an_operator.T_s_;

                // TODO(Etienne M): Make sure the accesses are sequential on
                // CPU.

                const auto team_count = CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta * CvCdDriver::kBlockExtentPhi;

                // Due to the inherent Cuda/HIP thread block limit.
                KOLIOP_ASSERT(team_count <= std::numeric_limits<int>::max());

                using PolicyType = Kokkos::TeamPolicy<
                    // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                    ExecutionSpace>;

                const auto a_policy = PolicyType{
                    an_execution_space,
                    static_cast<int>(team_count),
                    // NOTE: The thread block size and the Nmu benefit from being
                    // close. This is due to the TeamThreadMDRange which
                    // parallelizes only the left most index which is Nmu.
                    koliop::utility::Select<ExecutionSpace>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

                const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                    const OrdinalType team_index = a_team_handle.league_rank();

                    // NOTE: kBlockExtentXXX are constant, the compiler can replace
                    // the modulo with faster instructions.
                    const OrdinalType block_local_index_ir     = (team_index) % CvCdDriver::kBlockExtentR;
                    const OrdinalType block_local_index_itheta = (team_index / (CvCdDriver::kBlockExtentR)) % CvCdDriver::kBlockExtentTheta;
                    const OrdinalType block_local_index_iphi   = (team_index / (CvCdDriver::kBlockExtentR * CvCdDriver::kBlockExtentTheta)) /* % CvCdDriver::kBlockExtentPhi */;

                    if(!mask_coll(block_local_index_ir,
                                  block_local_index_itheta)) {
                        // NOTE: this return does not break the rest of the function
                        // as all the thread of the team will execute the same
                        // branch.
                        return;
                    }

                    const OffsetType flattened_index = team_index;

                    const OrdinalType batch_element_index = (flattened_index) % CvCdDriver::BatchTraitsType::kBatchSize;
                    const OrdinalType batch_index         = (flattened_index / (CvCdDriver::BatchTraitsType::kBatchSize)) /* % CvCdDriver::kBatchCount */;

                    const OrdinalType ir     = block_first_r + block_local_index_ir;
                    const OrdinalType itheta = block_first_theta + block_local_index_itheta;
                    const OrdinalType iphi   = block_first_phi + block_local_index_iphi;

                    const ComputeFloatType BoT = B_norm(local_domain_r_offset + ir,
                                                        local_domain_theta_offset + itheta) /
                                                 T_s(ir, itheta, iphi, species_index);

                    Kokkos::parallel_for(
                        // NOTE: The explicit Left iteration direction may not benefit
                        // the CPUs depending on how that is implemented.
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                  std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                    mu_extent, vpar_extent),
                        [&](auto imu, auto ivpar) {
                            const ComputeFloatType muBoT       = mug(imu) * BoT;
                            ComputeFloatType       f5d_s_value = 0.0;

                            for(OrdinalType ipol = 0; ipol < Constant::kMaximumLaguerrePolynomialDegree; ++ipol) {
                                f5d_s_value += rhs(batch_index, ivpar, ipol, batch_element_index) *
                                               DoComputeLaguerrePol(ipol, muBoT);
                            }

                            f5d_s(imu, ivpar, ir, itheta, iphi, species_index) = f5d_s_value;
                        });
                };

                Kokkos::parallel_for("DoComputeReconstruction",
                                     a_policy, kernel);
            }

        } // namespace detail


        ////////////////////////////////////////////////////////////////////////
        // CvCdDriver definition
        ////////////////////////////////////////////////////////////////////////

        template <OrdinalType block_extent_r,
                  OrdinalType block_extent_theta,
                  OrdinalType block_extent_phi,
                  typename BatchTraits>
        class CvCdDriver {
        public:
            static inline constexpr OrdinalType kBlockExtentR     = block_extent_r;
            static inline constexpr OrdinalType kBlockExtentTheta = block_extent_theta;
            static inline constexpr OrdinalType kBlockExtentPhi   = block_extent_phi;

            static inline constexpr OffsetType kFlattenedBlockExtent = kBlockExtentR * kBlockExtentTheta * kBlockExtentPhi;

            using BatchTraitsType = BatchTraits;

            static inline constexpr OffsetType kBatchCount = BatchTraitsType::BatchCount(kFlattenedBlockExtent);
            static_assert(kBatchCount * BatchTraitsType::kBatchSize == kFlattenedBlockExtent, "");

            using VectorKindType = koliop::blas::kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                                   Constant::kMaximumLaguerrePolynomialDegree>;

            using MatrixKindType = koliop::blas::kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                              Constant::kMaximumLaguerrePolynomialDegree>;

        protected:
            using MatrixSamplerType = MatrixSampler<typename MatrixKindType::ScalarType,
                                                    koliop::utility::View::InternalInputL1<const Flo64>>;

        public:
            CvCdDriver(OrdinalType                                                vpar_extent,
                       OrdinalType                                                species_extent,
                       bool                                                       collision_interspecies,
                       const koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                       const koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                       const koliop::utility::View::InternalInputL1<const Flo64>& vparg,
                       const koliop::utility::View::InternalInputL1<const Flo64>& coeff_intdvpar);

            template <typename Operator, // Templatized to avoid dependency loops
                      typename ExecutionSpace,
                      typename View6D>
            void Apply(const Operator&  an_operator,
                       ExecutionSpace   an_execution_space,
                       OrdinalType      block_first_r,
                       OrdinalType      block_first_theta,
                       OrdinalType      block_first_phi,
                       ComputeFloatType dt,
                       OrdinalType      species_index,
                       View6D&&         f5d_s);

        protected:
            // Miscellaneous numerical tooling

            Najl<CvCdDriver> najl_provider_;

            koliop::utility::View::MDL<bool[kBlockExtentR][kBlockExtentTheta]> mask_coll_;

            // NOTE: We know of an additional constant extent, the batch count
            // but we cannot exploit it due to kokkos enforcing that constant
            // extent must be placed after dynamic extent. vpar is non constant
            // and is placed after the batch count.
            typename MatrixKindType::ScalarType Najl_;
            typename MatrixKindType::ScalarType matrix_;
            typename VectorKindType::ScalarType alpha_coefficients_;
            typename VectorKindType::ScalarType rhs_;

            koliop::blas::BatchedBlockTridiagonalByBlockTRF<BatchTraitsType, Constant::kMaximumLaguerrePolynomialDegree>              factorizer_;
            koliop::blas::BatchedBlockTridiagonalByBlockTRSV<BatchTraitsType, Constant::kMaximumLaguerrePolynomialDegree>             solver_;
            koliop::blas::BatchedMatrixSamplingBlockTridiagonalByBlockMV<BatchTraitsType, Constant::kMaximumLaguerrePolynomialDegree> mv_;

            MatrixSamplerType LHS_matrix_sampler_;
            MatrixSamplerType RHS_matrix_sampler_;
        };


        ////////////////////////////////////////////////////////////////////////
        // CvCdDriver methods definition
        ////////////////////////////////////////////////////////////////////////

        template <OrdinalType block_extent_r,
                  OrdinalType block_extent_theta,
                  OrdinalType block_extent_phi,
                  typename BatchTraits>
        CvCdDriver<block_extent_r,
                   block_extent_theta,
                   block_extent_phi,
                   BatchTraits>::CvCdDriver(OrdinalType                                                vpar_extent,
                                            OrdinalType                                                species_extent,
                                            bool                                                       collision_interspecies,
                                            const koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                                            const koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                                            const koliop::utility::View::InternalInputL1<const Flo64>& vparg,
                                            const koliop::utility::View::InternalInputL1<const Flo64>& coeff_intdvpar)
            : najl_provider_{vpar_extent, species_extent,
                             collision_interspecies,
                             hat_As, hat_Zs, vparg}
            , mask_coll_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll")}
            , Najl_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "Najl"),
                    static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(vpar_extent)}
            , matrix_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         "matrix"),
                      static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(vpar_extent)}
            , alpha_coefficients_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                     "alpha_coefficients"),
                                  static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(vpar_extent)}
            , rhs_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                      "rhs"),
                   static_cast<std::size_t>(kBatchCount), static_cast<std::size_t>(vpar_extent)}
            , factorizer_{}
            , solver_{}
            , mv_{}
            , LHS_matrix_sampler_{Najl_,
                                  coeff_intdvpar}
            , RHS_matrix_sampler_{Najl_,
                                  coeff_intdvpar} {
            // EMPTY
        }

        template <OrdinalType block_extent_r,
                  OrdinalType block_extent_theta,
                  OrdinalType block_extent_phi,
                  typename BatchTraits>
        template <typename Operator, // Templatized to avoid dependency loops
                  typename ExecutionSpace,
                  typename View6D>
        void CvCdDriver<block_extent_r,
                        block_extent_theta,
                        block_extent_phi,
                        BatchTraits>::Apply(const Operator&  an_operator,
                                            ExecutionSpace   an_execution_space,
                                            OrdinalType      block_first_r,
                                            OrdinalType      block_first_theta,
                                            OrdinalType      block_first_phi,
                                            ComputeFloatType dt,
                                            OrdinalType      species_index,
                                            View6D&&         f5d_s) {

            // try {
            //     koliop::blas::detail::DoResidualTest(an_execution_space,
            //                                          factorizer_, solver_, mv_,
            //                                          matrix_, rhs_, alpha_coefficients_);
            // } catch(...) {
            // }

            // std::exit(42);

            const auto& vpar_extent  = an_operator.vpar_extent_;
            const auto& r_extent     = an_operator.r_extent_;
            const auto& theta_extent = an_operator.theta_extent_;
            const auto& phi_extent   = an_operator.phi_extent_;

            const auto& n_s   = an_operator.n_s_;
            const auto& T_s   = an_operator.T_s_;
            const auto& nu_ab = an_operator.nu_ab_;

            // TODO(Etienne M): Extent standardization global_domain, vs
            // local_domain vs block:
            // global_domain_[r|theta]_extent
            // local_domain_[r|theta]_offset
            // local_[mu|vpar|r|theta|phi|species]_extent
            // block_[r|theta|phi]_offset
            // block_[r|theta|phi]_extent

            // NOTE: Currently, we allocate memory to store the Najl coeff, the
            // alpha coeff, the matrix etc. What if we could compute the Alpha
            // coefs and overwrite the f5d_s with the matrix/alpha coef/najl ?

            // NOTE: If you have an issue with these assertions, checkout the
            // constant.h file and tune kBlockExtent[R|Theta|Phi].
            KOLIOP_ASSERT(r_extent % kBlockExtentR == 0);
            KOLIOP_ASSERT(theta_extent % kBlockExtentTheta == 0);
            KOLIOP_ASSERT(phi_extent % kBlockExtentPhi == 0);

            KOLIOP_ASSERT((block_first_r + kBlockExtentR) <= r_extent);
            KOLIOP_ASSERT((block_first_theta + kBlockExtentTheta) <= theta_extent);
            KOLIOP_ASSERT((block_first_phi + kBlockExtentPhi) <= phi_extent);

            // In the future, the time step could vary.
            LHS_matrix_sampler_.DeltaTime() = static_cast<ComputeFloatType>(-dt / 4.0);
            RHS_matrix_sampler_.DeltaTime() = static_cast<ComputeFloatType>(dt / 4.0);

            // NOTE: If mask_coll depends on r and theta, we could interchange
            // iphi with itheta and ir. that said, this call is so fast compared
            // to the rest that it does not seem useful.
            detail::DoPrepareCollisionMask<CvCdDriver>(an_operator,
                                                       an_execution_space,
                                                       block_first_r, block_first_theta,
                                                       mask_coll_);

            // NOTE: Skipping would create imbalance and because we
            // use large blocks, assuming the resolution is not too
            // large, we are pretty much assured that we get at
            // least one point we need to process.
            // if(all boolean in mask_coll are false) {
            //     continue;
            // }

            {
                // Projection of f on Laguerre Polynomials.
                detail::DoComputeAlphaProjection<CvCdDriver>(an_operator,
                                                             an_execution_space,
                                                             block_first_r, block_first_theta, block_first_phi,
                                                             species_index,
                                                             mask_coll_, f5d_s, alpha_coefficients_);

                {
                    // Computation of the N_i,a ^ jl = sum_b M_i,ab ^ jl.
                    najl_provider_.Compute(an_execution_space,
                                           block_first_r, block_first_theta, block_first_phi,
                                           species_index,
                                           n_s, T_s, nu_ab, Najl_);

                    // Computation of the RHS vector RHS_vec = (I + dt * T/4) * alpha
                    {
                        detail::DoPrepareMatrix<CvCdDriver>(an_operator,
                                                            an_execution_space,
                                                            LHS_matrix_sampler_,
                                                            matrix_);

                        // L*U = (I - T * dt/4)
                        factorizer_.Factorize(an_execution_space,
                                              matrix_.data(),
                                              kFlattenedBlockExtent, vpar_extent);
                    }

                    {
                        // rhs = (I + T * dt/4) * alpha(n)
                        mv_.Apply(an_execution_space,
                                  1.0,
                                  RHS_matrix_sampler_,
                                  kFlattenedBlockExtent, vpar_extent,
                                  alpha_coefficients_.data(),
                                  // Discards whatever garbage
                                  // rhs could contain.
                                  0.0,
                                  rhs_.data());
                    }

                    {
                        // Solve the problem (I - dt * T /4)*X = RHS_vec.
                        // rhs = alpha~
                        solver_.Solve(an_execution_space,
                                      matrix_.data(),
                                      kFlattenedBlockExtent, vpar_extent,
                                      rhs_.data());

                        std::swap(rhs_, alpha_coefficients_);

                        // Add the source term RHS = RHS + dt * S.
                        // rhs = alpha~~
                        detail::DoComputeCvCdSource<CvCdDriver>(an_operator,
                                                                an_execution_space,
                                                                block_first_r, block_first_theta, block_first_phi,
                                                                dt,
                                                                species_index,
                                                                Najl_, alpha_coefficients_);

                        // rhs = (I + T * dt/4) * alpha~~
                        mv_.Apply(an_execution_space,
                                  1.0,
                                  RHS_matrix_sampler_,
                                  kFlattenedBlockExtent, vpar_extent,
                                  alpha_coefficients_.data(),
                                  0.0,
                                  rhs_.data());

                        // (I - dt * T /4)*X = RHS_vec
                        // rhs = alpha(n+1)
                        solver_.Solve(an_execution_space,
                                      matrix_.data(),
                                      kFlattenedBlockExtent, vpar_extent,
                                      rhs_.data());

                        std::swap(rhs_, alpha_coefficients_);
                    }
                }

                // Reconstruct the normalized function distribution.
                detail::DoComputeReconstruction<CvCdDriver>(an_operator,
                                                            an_execution_space,
                                                            block_first_r, block_first_theta, block_first_phi,
                                                            species_index,
                                                            mask_coll_, alpha_coefficients_, f5d_s);
            }
        }

    } // namespace collision
} // namespace koliop

#endif
