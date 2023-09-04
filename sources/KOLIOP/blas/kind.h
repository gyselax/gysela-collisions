// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_KIND_H
#define KOLIOP_BLAS_KIND_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace blas {
        namespace kind {
            namespace vector {

                ////////////////////////////////////////////////////////////////////////
                // BatchedBlockByBlock definition
                ////////////////////////////////////////////////////////////////////////

                template <typename BatchTraits,
                          OrdinalType kSmallBlockExtent>
                struct BatchedBlockByBlock {
                public:
                    using BatchTraitsType = BatchTraits;

                    using value_type = typename BatchTraitsType::value_type;

                    using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

                public:
                    using ScalarType = Kokkos::View<value_type** [kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType,
                                                    Kokkos::MemoryTraits<Kokkos::Restrict>>;

                    using VectorType = Kokkos::View<typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType,
                                                    Kokkos::MemoryTraits<Kokkos::Restrict>>;

                    using ConstScalarType = Kokkos::View<const value_type** [kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType,
                                                         Kokkos::MemoryTraits<Kokkos::Restrict>>;

                    using ConstVectorType = Kokkos::View<const typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType,
                                                         Kokkos::MemoryTraits<Kokkos::Restrict>>;
                };

            } // namespace vector

            namespace matrix {

                ////////////////////////////////////////////////////////////////////////
                // BatchedBlockTridiagonalByBlock definition
                ////////////////////////////////////////////////////////////////////////

                template <typename BatchTraits,
                          OrdinalType kSmallBlockExtent>
                struct BatchedBlockTridiagonalByBlock {
                public:
                    using BatchTraitsType = BatchTraits;

                    using value_type = typename BatchTraitsType::value_type;

                    using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

                public:
                    /// We could make this more general but for now the code is assuming
                    /// this is fixed to 3.
                    static inline constexpr OrdinalType kDiagonalCount = 3;

                    using ScalarType = Kokkos::View<value_type** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType,
                                                    Kokkos::MemoryTraits<Kokkos::Restrict>>;

                    /// The batch count did not changed compared to ScalarType
                    /// but the batch potentially gets subdivided in vectors.
                    /// NOTE: We changed the View's type.
                    using VectorType = Kokkos::View<typename BatchTraitsType::ThreadVectorType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType,
                                                    Kokkos::MemoryTraits<Kokkos::Restrict>>;

                    using ConstScalarType = Kokkos::View<const value_type** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType,
                                                         Kokkos::MemoryTraits<Kokkos::Restrict>>;

                    using ConstVectorType = Kokkos::View<const typename BatchTraitsType::ThreadVectorType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType,
                                                         Kokkos::MemoryTraits<Kokkos::Restrict>>;
                };

            } // namespace matrix
        }     // namespace kind
    }         // namespace blas
} // namespace koliop

#endif
