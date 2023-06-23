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

            using ValueType = typename BatchTraitsType::ValueType;

            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

            using VectorKindType = kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                     kSmallBlockExtent>;

            // We do not require that the_A_matrix_sampler be a MatrixKindType
            // but we do require that it be *sampleable* similarly:
            // MatrixKindType{}(0,1,2,3,4,5); via operator()(...)
            using MatrixKindType = kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                kSmallBlockExtent>;

        public:
            BatchedMatrixSamplingBlockTridiagonalByBlockMV(ExecutionSpaceType an_execution_space);

            /// Computes the_y_vector = the_alpha_scalar * the_A_matrix(...) * the_x_vector +
            /// the_beta_scalar * the_y_vector.
            ///
            /// Because we assume the_A_matrix_sampler could be a View or an
            /// operator() we can not rely on the ThreadVectorType.
            ///
            template <typename MatrixSampler>
            void Apply(ValueType        the_alpha_scalar,
                       MatrixSampler&&  the_A_matrix_sampler,
                       OrdinalType      the_A_matrix_extent,
                       OrdinalType      the_M_matrix_extent,
                       const ValueType* the_x_vector,
                       ValueType        the_beta_scalar,
                       ValueType*       the_y_vector) const;

        protected:
            ExecutionSpaceType the_execution_space_;
        };


        ////////////////////////////////////////////////////////////////////////
        // BatchedMatrixSamplingBlockTridiagonalByBlockMV methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        BatchedMatrixSamplingBlockTridiagonalByBlockMV<BatchTraits,
                                                       kSmallBlockExtent>::BatchedMatrixSamplingBlockTridiagonalByBlockMV(ExecutionSpaceType an_execution_space)
            : the_execution_space_{an_execution_space} {
            // EMPTY
        }

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        template <typename MatrixSampler>
        void BatchedMatrixSamplingBlockTridiagonalByBlockMV<BatchTraits, kSmallBlockExtent>::Apply(ValueType        the_alpha_scalar,
                                                                                                   MatrixSampler&&  the_A_matrix_sampler,
                                                                                                   OrdinalType      the_A_matrix_extent,
                                                                                                   OrdinalType      the_M_matrix_extent,
                                                                                                   const ValueType* the_x_vector,
                                                                                                   ValueType        the_beta_scalar,
                                                                                                   ValueType*       the_y_vector) const {

            KOLIOP_ASSERT(the_x_vector != nullptr);
            KOLIOP_ASSERT(the_y_vector != nullptr);

            // the_A_matrix_extent must be divisible by
            // BatchTraitsType::kBatchSize. Else we need padding.
            KOLIOP_ASSERT(the_A_matrix_extent % BatchTraitsType::kBatchSize == 0);
            KOLIOP_ASSERT(the_M_matrix_extent > 0);

            const OrdinalType kBatchCount     = BatchTraitsType::BatchCount(the_A_matrix_extent);
            const OrdinalType kBigBlockExtent = the_M_matrix_extent;

            typename VectorKindType::ConstScalarType the_x_vector_view{reinterpret_cast<const typename BatchTraitsType::ValueType*>(the_x_vector),
                                                                       static_cast<std::size_t>(kBatchCount),
                                                                       static_cast<std::size_t>(kBigBlockExtent)};

            typename VectorKindType::ScalarType the_y_vector_view{reinterpret_cast<typename BatchTraitsType::ValueType*>(the_y_vector),
                                                                  static_cast<std::size_t>(kBatchCount),
                                                                  static_cast<std::size_t>(kBigBlockExtent)};

            // TODO(Etienne M): Check that we do not benefit from using
            // TeamGEMV This require a trick with vector threads.

            const auto the_kernel = KOKKOS_LAMBDA(auto the_batch_element_index, auto the_scalar_index, auto the_small_block_index, auto the_batch_index) {
                const auto the_vector_x_batch_element = Kokkos::subview(the_x_vector_view,
                                                                        the_batch_index,
                                                                        // TODO(Etienne M): These ALL() are constant, should we
                                                                        // hardcode them to facilitate the compiler's work ?
                                                                        Kokkos::ALL(), // kBigBlockExtent
                                                                        Kokkos::ALL(), // kSmallBlockExtent
                                                                        the_batch_element_index);

                ValueType the_dot_product{};

                // TODO(Etienne M): in practice, the_A_matrix_sampler samples
                // the Najl in a *transposed* way. That is, it could lead to
                // better accesses if we iterated faster on the
                // the_scalar_index than on j.

                // TODO(Etienne M): These reduction are not ideal for numerical
                // stability. Using more threads, we could do some kind of tree
                // like reduction with better properties.

                if(the_small_block_index > 0) {
                    // Every time, except for the first small block row.
                    for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                        the_dot_product += the_A_matrix_sampler(the_batch_index,
                                                                the_small_block_index,
                                                                0,
                                                                the_scalar_index, j,
                                                                the_batch_element_index) *
                                           the_vector_x_batch_element(the_small_block_index - 1, j);
                    }
                }

                // Every time.
                for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                    the_dot_product += the_A_matrix_sampler(the_batch_index,
                                                            the_small_block_index,
                                                            1,
                                                            the_scalar_index, j,
                                                            the_batch_element_index) *
                                       the_vector_x_batch_element(the_small_block_index, j);
                }

                if(the_small_block_index < (kBigBlockExtent - 1)) {
                    // Every time, except for the last small block row.
                    for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                        the_dot_product += the_A_matrix_sampler(the_batch_index,
                                                                the_small_block_index,
                                                                2,
                                                                the_scalar_index, j,
                                                                the_batch_element_index) *
                                           the_vector_x_batch_element(the_small_block_index + 1, j);
                    }
                }

                the_y_vector_view(the_batch_index,
                                  the_small_block_index,
                                  the_scalar_index,
                                  the_batch_element_index) *= the_beta_scalar;

                the_y_vector_view(the_batch_index,
                                  the_small_block_index,
                                  the_scalar_index,
                                  the_batch_element_index) += the_alpha_scalar * the_dot_product;
            };

            const auto a_policy = Kokkos::MDRangePolicy<ExecutionSpaceType,
                                                        Kokkos::Rank<4,
                                                                     // TODO(Etienne M): Tune blocking.
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{the_execution_space_,
                                                                                             {0, 0, 0, 0},
                                                                                             {BatchTraitsType::kBatchSize,
                                                                                              kSmallBlockExtent, kBigBlockExtent,
                                                                                              kBatchCount}};

            Kokkos::parallel_for("BatchedMatrixSamplingBlockTridiagonalByBlockMV::Apply",
                                 a_policy, the_kernel);
        }

    } // namespace blas
} // namespace koliop

#endif
