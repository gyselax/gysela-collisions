// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_CONSTANT_H
#define KOLIOP_COLLISION_CONSTANT_H

#include "KOLIOP/blas/batch.h"
#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace collisions {

        ////////////////////////////////////////////////////////////////////////
        // Constant definition
        ////////////////////////////////////////////////////////////////////////

        struct Constant {
        public:
            static inline constexpr OrdinalType kMaximumLaguerrePolynomialDegree = 3;

            // TODO(Etienne M): better name
            static inline constexpr OrdinalType kIndexMax = 2 * (kMaximumLaguerrePolynomialDegree - 1);

            using HostExecutionSpaceType   = Kokkos::DefaultHostExecutionSpace;
            using DeviceExecutionSpaceType = Kokkos::DefaultExecutionSpace;

            using BatchTraitsType = koliop::blas::BatchTraits<ComputeFloatType,
                                                              DeviceExecutionSpaceType>;

            /// Used to handle cases where the operator is to work on very small
            /// spatial domain (say, 1x1x1).
            ///
            using FakeBatchTraitsType = koliop::blas::FakeBatchTraits<ComputeFloatType,
                                                                      DeviceExecutionSpaceType>;

            // Aka, HIP/Cuda stream count. Bigger means more memory used but
            // also, potentially faster. Trade memory and inconvenience for
            // speed.
            // TODO(Etienne M): Depending on the hardware, GPU/CPU, and between
            // GPUs, this value could be increased or reduced. Using future
            // GPUs, we could benefit from a higher value.
            // Bigger blocking needs less streams.
            // NOTE: It may be that the block size is too small to use the whole
            // GPU's resources on every kernels. By using streams we can fill
            // the gapes left by the small kernels.

#if defined(KOLIOP_BACKEND_IS_GPU_LIKE)
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount = 1;

            /// kDiscardedStreamCount is used to define how many stream to
            /// discard. The reason for this shady business is that when Koliop
            /// is used with HIP the stream mapping to hardware queue has a
            /// weird behavior, see, on MI250X with 4 hardware queues:
            /// https://github.com/ROCm/hip/issues/3769
            /// If two streams share the same HW queue, the kernels are
            /// serialized. So to ensure we are in linear "stream mapping
            /// regime", we "consume" at least <hardware queue count> streams.
            ///
            static inline constexpr OrdinalType kDiscardedStreamCount = /* GPU_MAX_HW_QUEUES * 2 */ 8;
#else
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount = 1;
            static inline constexpr OrdinalType kDiscardedStreamCount        = 0;
#endif
            static inline constexpr OrdinalType kDefaultExecutionSpacePoolIndex = kDisjointExecutionSpaceCount - 1;
            static_assert((kDisjointExecutionSpaceCount >= 1) && (kDefaultExecutionSpacePoolIndex < kDisjointExecutionSpaceCount),
                          "");
        };

    } // namespace collisions
} // namespace koliop

#endif
