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

            // Aka, HIP/Cuda stream count.
            // TODO(Etienne M): Depending on the hardware, GPU/CPU, and between
            // GPUs, this value could be increased or reduced. Using future
            // GPUs, we could benefit from a higher value.
            static inline constexpr OrdinalType kDisjointExecutionSpaceCount    = 4;
            static inline constexpr OrdinalType kDefaultExecutionSpacePoolIndex = 0;
            static_assert(kDefaultExecutionSpacePoolIndex < kDisjointExecutionSpaceCount, "");

            // TODO(Etienne M): Note we should probably not use constant for the
            // blocking. Indeed, on CPU we benefit from a small block and on GPU
            // we would rather to everything at once if possible. This leads us
            // to the dilemma where on CPU due to the small block size, we can
            // afford a constraint on the simulation through
            // kBlockExtent[R, Theta, Phi] but on GPU, this constraints would be
            // unaffordable. On GPU we need dynamic kBlockExtent[R, Theta, Phi].
            // This cost use a *lot* (still has to be measured) in modulo
            // operations.

            // TODO(Etienne M): Tune that size or rework batching. If we can not
            // have larger block, we have to check if kokkos supports
            // distributing on vector/teams.

            // NOTE: The amount of memory used during the collision operator is
            // at most approximately:
            // 8*(Nspace*Nvpar*Nmu + 2*3*3*3*Nspace*Nvpar + 1*3*Nspace*Nvpar)
            // fd5                   Matrix                 RHS

#if defined(KOKKOS_ENABLE_HIP)
            static inline constexpr OrdinalType kBlockExtentR     = 128;
            static inline constexpr OrdinalType kBlockExtentTheta = 32;
            static inline constexpr OrdinalType kBlockExtentPhi   = 4;

                // static inline constexpr OrdinalType kBlockExtentR     = 32;
                // static inline constexpr OrdinalType kBlockExtentTheta = 16;
                // static inline constexpr OrdinalType kBlockExtentPhi   = 2;

#elif defined(KOKKOS_ENABLE_OPENMP)
            static inline constexpr OrdinalType kBlockExtentR     = 16;
            static inline constexpr OrdinalType kBlockExtentTheta = 16;
            static inline constexpr OrdinalType kBlockExtentPhi   = 1;
#else
    #error "Unsupported architecture."
#endif
        };

    } // namespace collision
} // namespace koliop

#endif
