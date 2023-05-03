// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_MV_H
#define KOLIOP_BLAS_MV_H

#include "KOLIOP/internal/configuration.h"

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

            using ValueType          = typename BatchTraitsType::ValueType;
            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

        public:
            using ScalarVectorType = Kokkos::View<ValueType** [kSmallBlockExtent][BatchTraitsType::kBatchSize],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpaceType>;

            using VectorVectorType = Kokkos::View<typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpaceType>;

            using ConstVectorVectorType = Kokkos::View<const typename BatchTraitsType::ThreadVectorType** [kSmallBlockExtent][BatchTraitsType::kVectorPerBatch],
                                                       Kokkos::LayoutRight,
                                                       ExecutionSpaceType>;

        protected:
            // using ConcurrencyPolicy = KokkosBatched::Mode::Serial;
            using CachingPolicy = KokkosBatched::Algo::Level3::Unblocked;

        public:
            BatchedMatrixSamplingBlockTridiagonalByBlockMV(ExecutionSpaceType an_execution_space);

            /// @brief Computes the_y_vector = the_alpha_scalar * the_A_matrix(...) * the_x_vector +
            /// the_beta_scalar * the_y_vector
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

            ConstVectorVectorType the_x_vector_view{reinterpret_cast<const typename BatchTraitsType::ThreadVectorType*>(the_x_vector),
                                                    static_cast<std::size_t>(kBatchCount),
                                                    static_cast<std::size_t>(kBigBlockExtent)};

            VectorVectorType the_y_vector_view{reinterpret_cast<typename BatchTraitsType::ThreadVectorType*>(the_y_vector),
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

                typename BatchTraitsType::ThreadVectorType the_dot_product{};

                // TODO(Etienne M): in practice, the_A_matrix_sampler samples
                // the Nia in a transposed order. That is, it could lead to
                // better access if we iterated faster on the
                // the_scalar_index than on j.

                if(the_small_block_index > 0) {
                    // Every time, except for the first small block row.
                    for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                        the_dot_product += the_A_matrix_sampler.UpperDiagonal(the_batch_index,
                                                                              the_small_block_index,
                                                                              the_scalar_index, j,
                                                                              the_batch_element_index) *
                                           the_vector_x_batch_element(the_small_block_index - 1, j);
                    }
                }

                // Every time.
                for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                    the_dot_product += the_A_matrix_sampler.MiddleDiagonal(the_batch_index,
                                                                           the_small_block_index,
                                                                           the_scalar_index, j,
                                                                           the_batch_element_index) *
                                       the_vector_x_batch_element(the_small_block_index, j);
                }

                if(the_small_block_index < (kBigBlockExtent - 1)) {
                    // Every time, except for the last small block row.
                    for(OrdinalType j = 0; j < kSmallBlockExtent; ++j) {
                        the_dot_product += the_A_matrix_sampler.UpperDiagonal(the_batch_index,
                                                                              the_small_block_index,
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
                                                                                             {BatchTraitsType::kVectorPerBatch,
                                                                                              kSmallBlockExtent, kBigBlockExtent,
                                                                                              kBatchCount}};

            Kokkos::parallel_for("BatchedMatrixSamplingBlockTridiagonalByBlockMV::Apply",
                                 a_policy, the_kernel);
        }

    } // namespace blas
} // namespace koliop

#endif
