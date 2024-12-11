// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_MV_H
#define KOLIOP_BLAS_MV_H

#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // BatchedMatrixSamplingBlockTridiagonalByBlockMV definition
        ////////////////////////////////////////////////////////////////////////

        /// See BatchedBlockTridiagonalByBlockTRF for more details on the data
        /// layout.
        ///
        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        class BatchedMatrixSamplingBlockTridiagonalByBlockMV {
        public:
            using BatchTraitsType = BatchTraits;

            using value_type = typename BatchTraitsType::value_type;

            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

            using VectorKindType = kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                     kSmallBlockExtent>;

            // We do not require that A_matrix_sampler be a MatrixKindType
            // but we do require that it be *sampleable* similarly:
            // MatrixKindType{}(0,1,2,3,4,5); via operator()(...)
            using MatrixKindType = kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                kSmallBlockExtent>;

        public:
            /// Computes y_vector = alpha_scalar * A_matrix(...) * x_vector +
            /// beta_scalar * y_vector.
            ///
            /// Because we assume A_matrix_sampler could be a View or an
            /// operator() we can not rely on the ThreadVectorType.
            ///
            template <typename MatrixSampler>
            void Apply(ExecutionSpaceType an_execution_space,
                       value_type         alpha_scalar,
                       MatrixSampler&&    A_matrix_sampler,
                       OrdinalType        A_matrix_extent,
                       OrdinalType        M_matrix_extent,
                       const value_type*  x_vector,
                       value_type         beta_scalar,
                       value_type*        y_vector) const;

        protected:
        };


        ////////////////////////////////////////////////////////////////////////
        // BatchedMatrixSamplingBlockTridiagonalByBlockMV methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        template <typename MatrixSampler>
        void BatchedMatrixSamplingBlockTridiagonalByBlockMV<BatchTraits, kSmallBlockExtent>::Apply(ExecutionSpaceType an_execution_space,
                                                                                                   value_type         alpha_scalar,
                                                                                                   MatrixSampler&&    A_matrix_sampler,
                                                                                                   OrdinalType        A_matrix_extent,
                                                                                                   OrdinalType        M_matrix_extent,
                                                                                                   const value_type*  x_vector,
                                                                                                   value_type         beta_scalar,
                                                                                                   value_type*        y_vector) const {

            KOLIOP_ASSERT(x_vector != nullptr);
            KOLIOP_ASSERT(y_vector != nullptr);

            // A_matrix_extent must be divisible by
            // BatchTraitsType::kBatchSize. Else we need padding.
            KOLIOP_ASSERT(A_matrix_extent % BatchTraitsType::kBatchSize == 0);
            KOLIOP_ASSERT(M_matrix_extent > 0);

            const OrdinalType kBatchCount     = BatchTraitsType::BatchCount(A_matrix_extent);
            const OrdinalType kBigBlockExtent = M_matrix_extent;

            typename VectorKindType::ConstScalarType x_vector_view{reinterpret_cast<const typename BatchTraitsType::value_type*>(x_vector),
                                                                   static_cast<std::size_t>(kBatchCount),
                                                                   static_cast<std::size_t>(kBigBlockExtent)};

            typename VectorKindType::ScalarType y_vector_view{reinterpret_cast<typename BatchTraitsType::value_type*>(y_vector),
                                                              static_cast<std::size_t>(kBatchCount),
                                                              static_cast<std::size_t>(kBigBlockExtent)};

            // TODO(Etienne M): Check that we do not benefit from using
            // TeamGEMV This require a trick with vector threads.

            const auto kernel = KOKKOS_LAMBDA(OrdinalType batch_element_index, OrdinalType scalar_index, OrdinalType small_block_index, OrdinalType batch_index) {
                const auto vector_x_batch_element = Kokkos::subview(x_vector_view,
                                                                    batch_index,
                                                                    // TODO(Etienne M): These ALL() are constant, should we
                                                                    // hardcode them to facilitate the compiler's work ?
                                                                    Kokkos::ALL(), // kBigBlockExtent
                                                                    Kokkos::ALL(), // kSmallBlockExtent
                                                                    batch_element_index);

                value_type dot_product{};

                // TODO(Etienne M): These reduction are not ideal for numerical
                // stability. Using more threads, we could do some kind of tree
                // like reduction with better properties.

                if(small_block_index > 0) {
                    // Every time, except for the first small block row.
                    for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                        dot_product += A_matrix_sampler(batch_index,
                                                        small_block_index,
                                                        0,
                                                        scalar_index, j,
                                                        batch_element_index) *
                                       vector_x_batch_element(small_block_index - 1, j);
                    }
                }

                // Every time.
                for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                    dot_product += A_matrix_sampler(batch_index,
                                                    small_block_index,
                                                    1,
                                                    scalar_index, j,
                                                    batch_element_index) *
                                   vector_x_batch_element(small_block_index, j);
                }

                if(static_cast<OrdinalType>(small_block_index) < (kBigBlockExtent - 1)) {
                    // Every time, except for the last small block row.
                    for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                        dot_product += A_matrix_sampler(batch_index,
                                                        small_block_index,
                                                        2,
                                                        scalar_index, j,
                                                        batch_element_index) *
                                       vector_x_batch_element(small_block_index + 1, j);
                    }
                }

                y_vector_view(batch_index,
                              small_block_index,
                              scalar_index,
                              batch_element_index) *= beta_scalar;

                y_vector_view(batch_index,
                              small_block_index,
                              scalar_index,
                              batch_element_index) += alpha_scalar * dot_product;
            };

            const auto a_policy = Kokkos::MDRangePolicy<
                // Kokkos::LaunchBounds<256, 1>, Kokkos::IndexType<OrdinalType>,
                ExecutionSpaceType,
                Kokkos::Rank<4,
                             // TODO(Etienne M): Tune blocking.
                             Kokkos::Iterate::Left,
                             Kokkos::Iterate::Left>>{an_execution_space,
                                                     {0, 0, 0, 0},
                                                     {BatchTraitsType::kBatchSize,
                                                      kSmallBlockExtent, kBigBlockExtent,
                                                      kBatchCount}};

            Kokkos::parallel_for("BatchedMatrixSamplingBlockTridiagonalByBlockMV::Apply",
                                 a_policy, kernel);
        }

    } // namespace blas
} // namespace koliop

#endif
