// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_OPERATOR_H
#define KOLIOP_COLLISION_OPERATOR_H

#include "KOLIOP/blas/batch.h"
#include "KOLIOP/blas/gemv.h"
#include "KOLIOP/blas/getrf.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Operator definition
        ////////////////////////////////////////////////////////////////////////

        /// TODO(Etienne M): INDEX We have a dilemma. CPU allows for large
        /// temporary, per thread, buffers. GPU do not (at least for LDS). Due
        /// to caches hierarchy, CPU like per thread, contiguous memory
        /// accesses. GPU do not because this kind of access leads to wastes in
        /// the memory transactions (notion of un-coalesced accesses). Using
        /// Kokkos, we hope to attain decent performance on both GPU and CPU.
        ///
        /// Assuming the following example, all arrays with layout left
        /// (Fortran).
        ///
        ///     collapse(3)
        ///     for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///         for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///             for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                 const Flo64 a = A(ir, itheta);
        ///                 const Flo64 b = B(ir, itheta, iphi);
        ///                 for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///                     const Flo64 c = C(ivpar);
        ///                     for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                         const Flo64 d = D(imu);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// Collapsing on the first 3 loops would lead to decent performance on
        /// CPU. On GPU though, this would be crappy as each thread would do
        /// linear access in C, D and E. Access in C and D are *ok* that bad,
        /// because, out of one memory transaction, conceptualize it as a cache
        /// line, only 8 bytes will be kept but the next fetch will be likely to
        /// hit the cache. Except that on MI250X cards, we have very few cache
        /// (8 Mio per GCD!). Access in E are bad because each thread gets a
        /// different ir value which means, thread n+1 will access memory in a
        /// stride of mumax*vparmax (strongly un-coalesced).
        ///
        /// We could try to inter-exchange the loops but this would then work
        /// badly on CPU.
        ///
        ///     collapse(3)
        ///     for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///         for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///             for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                 const Flo64 a = A(ir, itheta);
        ///                 const Flo64 c = C(ivpar);
        ///                 const Flo64 d = D(imu);
        ///                 for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///                     for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                         const Flo64 b = B(ir, itheta, iphi);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// In the end, the most versatile is to collapse everything, do
        /// redundant computation on GPU but dont waste what you bring from
        /// global memory and hope that the compiler will optimize the redundant
        /// loads and computations computation on CPU.
        ///
        ///     collapse(5)
        ///     for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///         for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///             for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                 for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///                     for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                         const Flo64 a = A(ir, itheta);
        ///                         const Flo64 c = C(ivpar);
        ///                         const Flo64 d = D(imu);
        ///                         const Flo64 b = B(ir, itheta, iphi);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// TODO(Etienne M): Convert all these indicies to half open ranges
        /// counter.
        ///
        class Operator {
        public:
            static inline constexpr OrdinalType kMaximumLaguerrePolynomialDegree = 3;

        public:
            using BatchTraitsType = koliop::blas::BatchTraits<ComputeFloatType,
                                                              Kokkos::DefaultExecutionSpace>;

            using BatchedSolverType = koliop::blas::BatchedGETRFBlockTridiagonalByBlock<BatchTraitsType,
                                                                                        kMaximumLaguerrePolynomialDegree>;

            using BatchedGEMVType = koliop::blas::MatrixSamplingBatchedGEMVBlockTridiagonalByBlock<BatchTraitsType,
                                                                                                   kMaximumLaguerrePolynomialDegree>;

            // TODO(Etienne M): Tune that size or rework batching
            static inline constexpr OrdinalType kBlockExtentR     = 16;
            static inline constexpr OrdinalType kBlockExtentTheta = 1;
            static inline constexpr OrdinalType kBlockExtentPhi   = 1;

            static inline constexpr OffsetType kFlattenedBlockExtent = kBlockExtentR * kBlockExtentTheta * kBlockExtentPhi;

            static inline constexpr OffsetType kBatchCount = kFlattenedBlockExtent / BatchTraitsType::kBatchSize;
            static_assert(kBatchCount * BatchTraitsType::kBatchSize == kFlattenedBlockExtent, "");

        public:
            Operator(uint64_t     pSpecies_id,
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
                     Flo64        dvpar,
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
                     const Flo64* Bstar_s);

            void ComputeMoments(Flo64* f5d_s);

            void PrecomputeNu();

            void ComputeMaxwellianEvolution(Flo64* f5d_s);

            void ComputeQparOnNT(const Flo64* f5d_s);

            void ComputeCpar(Flo64* f5d_s) const;

            void ComputeMaxwellianUnshiftedLoc(Flo64*   f5d_s,
                                               uint32_t choice) const;

            void Apply(Flo64* f5d_s) const;

            void ComputeDensityCorrection(Flo64* f5d_s) const;

            void ComputeUparTCorrection(Flo64* f5d_s) const;

        public:
            // Constant, externally computed values

            // TODO(Etienne M): Cleanup unused values
            OrdinalType      pSpecies_id_;
            OrdinalType      mu_id_;
            OrdinalType      the_mu_extent_;
            OrdinalType      the_vpar_extent_;
            OrdinalType      the_r_extent_;
            OrdinalType      the_theta_extent_;
            OrdinalType      the_phi_extent_;
            OrdinalType      the_species_extent_;
            ComputeFloatType R0_;
            bool             collision_interspecies_;
            bool             SOL_;
            bool             LIM_;
            OrdinalType      ir_SOL_separatrix_;
            ComputeFloatType dt_;
            // TODO(Etienne M): dvpar is already present in gysela. Because we
            // assume (for now) an uniform vpar discretization, vparg could be
            // removed in favor of a closed formula. To avoid changing the
            // interface we recompute it
            // When we start using a sampler, remove this value.
            ComputeFloatType dvpar_;

            koliop::utility::View::MDD2<const ComputeFloatType> comb_mat_; // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD1<const ComputeFloatType> hat_As_;   // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD1<const ComputeFloatType> hat_Zs_;   // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD1<const ComputeFloatType> mug_;
            // NOTE: In practice the grid is uniform in vparg so you
            // could compute vparg(n) = vparg(0) + dvpar * n.
            // TODO(Etienne M): replace vparg by a sampler and it should be
            // mostly transparent to the user
            koliop::utility::View::MDD1<const ComputeFloatType> vparg_;
            koliop::utility::View::MDD1<const ComputeFloatType> coeff_intdmu_;
            koliop::utility::View::MDD1<const ComputeFloatType> coeff_intdvpar_;
            koliop::utility::View::MDD1<const ComputeFloatType> rg_;
            koliop::utility::View::MDD1<const ComputeFloatType> nustar0_r_;
            koliop::utility::View::MDD1<const ComputeFloatType> q_r_;
            koliop::utility::View::MDD1<const ComputeFloatType> mask_buffer_r_;
            koliop::utility::View::MDD2<const ComputeFloatType> mask_LIM_;
            koliop::utility::View::MDD2<const ComputeFloatType> B_norm_;
            koliop::utility::View::MDD4<const ComputeFloatType> Bstar_s_;

            // Constant, internally computed values

            koliop::utility::View::MDD2<ComputeFloatType> C_j_;  // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType> C_jl_; // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType> nustar_coeff_;

            // Variable, internally computed values

            koliop::utility::View::MDD4<ComputeFloatType> n_s_;
            koliop::utility::View::MDD4<ComputeFloatType> U_par_s_;
            koliop::utility::View::MDD4<ComputeFloatType> T_s_;
            koliop::utility::View::MDD5<ComputeFloatType> nu_ab_;
            koliop::utility::View::MDD4<ComputeFloatType> q_par_on_nT_s_;

            // Miscellaneous tooling

            BatchedSolverType the_alpha_solver_;
            BatchedGEMVType   the_gemv_;
        };

    } // namespace collision
} // namespace koliop

#endif
