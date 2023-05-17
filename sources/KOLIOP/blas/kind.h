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

                    using ValueType = typename BatchTraitsType::ValueType;

                    using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

                public:
                    using ScalarType = Kokkos::View<ValueType** [kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType>;

                    using VectorType = Kokkos::View<typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType>;

                    using ConstScalarType = Kokkos::View<const ValueType** [kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType>;

                    using ConstVectorType = Kokkos::View<const typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType>;
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

                    using ValueType = typename BatchTraitsType::ValueType;

                    using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

                public:
                    /// We could make this more general but for now the code is assuming
                    /// this is fixed to 3.
                    static inline constexpr OrdinalType kDiagonalCount = 3;

                    using ScalarType = Kokkos::View<ValueType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType>;

                    /// The batch count did not changed compared to ScalarType
                    /// but the batch potentially gets subdivided in vectors.
                    /// NOTE: We changed the View's type.
                    using VectorType = Kokkos::View<typename BatchTraitsType::ThreadVectorType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                    Kokkos::LayoutRight,
                                                    ExecutionSpaceType>;

                    using ConstScalarType = Kokkos::View<const ValueType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType>;

                    using ConstVectorType = Kokkos::View<const typename BatchTraitsType::ThreadVectorType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                         Kokkos::LayoutRight,
                                                         ExecutionSpaceType>;
                };

            } // namespace matrix
        }     // namespace kind
    }         // namespace blas
} // namespace koliop

#endif
