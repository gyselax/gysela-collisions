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

            // Aka, HIP/Cuda stream count. Bigger means more memory used but
            // also, potentially faster. Trade memory and inconvenience for
            // speed.
            // TODO(Etienne M): Depending on the hardware, GPU/CPU, and between
            // GPUs, this value could be increased or reduced. Using future
            // GPUs, we could benefit from a higher value.
            // Bigger blocking needs less streams ad inversely.

#if defined(KOKKOS_ENABLE_HIP)
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount = 4;
#elif defined(KOKKOS_ENABLE_OPENMP)
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount = 1;
#endif
            static inline constexpr OrdinalType kDefaultExecutionSpacePoolIndex = kDisjointExecutionSpaceCount - 1;
            static_assert((kDisjointExecutionSpaceCount >= 1) && (kDefaultExecutionSpacePoolIndex < kDisjointExecutionSpaceCount),
                          "");

            // TODO(Etienne M): Note we should probably not use constant for the
            // blocking. Indeed, on CPU we benefit from a small block and on GPU
            // we would rather to everything at once if possible. This leads us
            // to the dilemma where on CPU due to the small block size, we can
            // afford a constraint on the simulation through
            // kBlockExtent[R, Theta, Phi] but on GPU, this constraints would be
            // unaffordable. On GPU we need dynamic kBlockExtent[R, Theta, Phi].

            // NOTE: The amount of memory used during the collision operator is
            // at most approximately:
            // 8*(Nspace*Nvpar*Nmu + 2*3*3*3*Nspace*Nvpar + 1*3*Nspace*Nvpar)
            // fd5                   Matrix                 RHS

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_OPENMP)
            static inline constexpr std::size_t kPresetBlockExtentCount                        = 22;
            static inline constexpr OrdinalType kPresetBlockExtent[kPresetBlockExtentCount][3] = {{256, 128, 2},
                                                                                                  {128, 256, 2},
                                                                                                  {256, 128, 1},
                                                                                                  {128, 256, 1},
                                                                                                  {128, 128, 2},
                                                                                                  {128, 128, 1},
                                                                                                  {128, 64, 2},
                                                                                                  {64, 128, 2},
                                                                                                  {128, 64, 1},
                                                                                                  {128, 32, 2},
                                                                                                  {64, 128, 1},
                                                                                                  {64, 64, 2},
                                                                                                  {32, 128, 2},
                                                                                                  {128, 32, 1},
                                                                                                  {64, 64, 1},
                                                                                                  {64, 32, 2},
                                                                                                  {32, 128, 1},
                                                                                                  {32, 64, 2},
                                                                                                  {64, 32, 1},
                                                                                                  {32, 64, 1},
                                                                                                  {32, 32, 2},
                                                                                                  {32, 32, 1}};
#else
    #error "Unsupported architecture."
#endif
        };

    } // namespace collision
} // namespace koliop

#endif
