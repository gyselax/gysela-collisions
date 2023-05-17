// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_CONSTANT_H
#define KOLIOP_COLLISION_CONSTANT_H

#include "KOLIOP/blas/batch.h"
#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Constant definition
        ////////////////////////////////////////////////////////////////////////

        struct Constant {
        public:
            static inline constexpr OrdinalType kMaximumLaguerrePolynomialDegree = 3;

            // TODO(Etienne M): better name
            static inline constexpr OrdinalType kIndexMax = 2 * (kMaximumLaguerrePolynomialDegree - 1);

            using DeviceExecutionSpaceType = Kokkos::DefaultExecutionSpace;

            using BatchTraitsType = koliop::blas::BatchTraits<ComputeFloatType,
                                                              DeviceExecutionSpaceType>;

            // TODO(Etienne M): Note we should probably not use constant for the
            // blocking. Indeed, on CPU we benefit from a small block and on GPU
            // we would rather to everything at once if possible. This leads us
            // to the dilemma where on CPU due to the small block size, we can
            // afford a constraint on the simulation through
            // kBlockExtent[R, Theta, Phi] but on GPU, this constraints would be
            // unaffordable. On GPU we need dynamic kBlockExtent[R, Theta, Phi].
            // This cost use a *lot* (still has to be measured) in modulo
            // operations.

            // TODO(Etienne M): Tune that size or rework batching
            static inline constexpr OrdinalType kBlockExtentR     = 16;
            static inline constexpr OrdinalType kBlockExtentTheta = 1;
            static inline constexpr OrdinalType kBlockExtentPhi   = 1;

            static inline constexpr OffsetType kFlattenedBlockExtent = kBlockExtentR * kBlockExtentTheta * kBlockExtentPhi;

            static inline constexpr OffsetType kBatchCount = BatchTraitsType::BatchCount(kFlattenedBlockExtent);
            static_assert(kBatchCount * BatchTraitsType::kBatchSize == kFlattenedBlockExtent, "");

        public:
            using VectorKindType = koliop::blas::kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                                   kMaximumLaguerrePolynomialDegree>;

            using MatrixKindType = koliop::blas::kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                              kMaximumLaguerrePolynomialDegree>;
        };

    } // namespace collision
} // namespace koliop

#endif
