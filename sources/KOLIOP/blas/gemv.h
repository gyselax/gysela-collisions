// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_GEMV_H
#define KOLIOP_BLAS_GEMV_H

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // MatrixSamplingBatchedGEMVBlockTridiagonalByBlock definition
        ////////////////////////////////////////////////////////////////////////

        ///
        ///
        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        class MatrixSamplingBatchedGEMVBlockTridiagonalByBlock {
        public:
            using BatchTraitsType = BatchTraits;

            using ValueType          = typename BatchTraitsType::ValueType;
            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

        public:
            /// We could make this more general but for now the code is assuming
            /// this is fixed to 3.
            static inline constexpr OrdinalType kDiagonalCount = 3;

            using ScalarVectorType = Kokkos::View<ValueType** [kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpaceType>;

            using VectorVectorType = Kokkos::View<typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpaceType>;

        protected:
            // using ConcurrencyPolicy = KokkosBatched::Mode::Serial;
            using CachingPolicy = KokkosBatched::Algo::Level3::Unblocked;

        public:
            MatrixSamplingBatchedGEMVBlockTridiagonalByBlock(ExecutionSpaceType an_execution_space);

            template <typename MatrixSampler>
            void Apply(MatrixSampler&& a_sampler,
                       ValueType*      the_vector) const;

        protected:
            ExecutionSpaceType the_execution_space_;
        };


        ////////////////////////////////////////////////////////////////////////
        // MatrixSamplingBatchedGEMVBlockTridiagonalByBlock methods definition
        ////////////////////////////////////////////////////////////////////////


    } // namespace blas
} // namespace koliop

#endif
