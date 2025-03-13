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
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount = 2;
#else
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount = 1;
#endif
            static inline constexpr OrdinalType kDefaultExecutionSpacePoolIndex = kDisjointExecutionSpaceCount - 1;
            static_assert((kDisjointExecutionSpaceCount >= 1) && (kDefaultExecutionSpacePoolIndex < kDisjointExecutionSpaceCount),
                          "");

            // NOTE: The amount of memory used internally by the operator
            // depends on the number of disjoint execution space allowing
            // asynchronous blocked CvCd computation and on the maximum block
            // size that can be used to fill the rank local domain.
            // For a fixed kDisjointExecutionSpaceCount lowering the block size
            // wont have an impact on the memory consumption until all the
            // kDisjointExecutionSpaceCount have at least a block each to
            // process. Else, lowering either kDisjointExecutionSpaceCount (down
            // to 1) and lowering the block size can help reduce the memory
            // consumption.

            /// {Tor1, Tor2, Tor3} block extents.
            ///
            static inline constexpr OrdinalType kPresetBlockExtent[][3] = {
                // Nspace = 65536
                // {256, 128, 2},
                // {128, 256, 2},
                // Nspace = 32768
                // {256, 128, 1},
                // {128, 256, 1},
                // {128, 128, 2},
                // Nspace = 16384
                // {128, 128, 1},
                // {128, 64, 2},
                // {128, 32, 4},
                // {64, 128, 2},
                // Nspace = 8192
                {128, 64, 1},
                {128, 32, 2},
                {64, 128, 1},
                {64, 64, 2},
                {32, 128, 2},
                // Nspace = 4096
                {128, 32, 1},
                {64, 64, 1},
                {64, 32, 2},
                {32, 128, 1},
                {32, 64, 2},
                // Nspace = 2048
                {64, 32, 1},
                {32, 64, 1},
                {32, 32, 2},
                // Nspace = 1024
                {32, 32, 1},
                // Nspace = 1
                // NOTE: Degraded operation performance is expected for 1x1x1
                // blocks.
                {1, 1, 1}};

            static inline constexpr std::size_t kPresetBlockExtentCount = sizeof(kPresetBlockExtent) /
                                                                          sizeof(kPresetBlockExtent[0]);
        };

    } // namespace collision
} // namespace koliop

#endif
