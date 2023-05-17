// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_TRSV_H
#define KOLIOP_BLAS_TRSV_H

#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRSV definition
        ////////////////////////////////////////////////////////////////////////

        /// @brief See BatchedBlockTridiagonalByBlockTRSV for a detailed matrix
        /// layout explanation.
        /// @tparam BatchTraits
        /// @tparam kSmallBlockExtent
        template <typename BatchTraits,
                  // TODO(Etienne M): See if kFlattenedBlockExtent which is
                  // constexpr, can help improve the performance when used as an
                  // template argument. kBigBlockExtent = kFlattenedBlockExtent.
                  // This would remove the_A_matrix_extent argument.
                  OrdinalType kSmallBlockExtent>
        class BatchedBlockTridiagonalByBlockTRSV {
        public:
            using BatchTraitsType = BatchTraits;

            using ValueType = typename BatchTraitsType::ValueType;

            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

            using VectorKindType = kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                     kSmallBlockExtent>;

            using MatrixKindType = kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                kSmallBlockExtent>;

        protected:
        public:
            BatchedBlockTridiagonalByBlockTRSV(ExecutionSpaceType an_execution_space);

            /// @brief TRSV, solve Ax = b where A is a LU factorization output
            /// by BatchedBlockTridiagonalByBlockTRF::Factorize().
            /// @param the_rhs In place resolution. the_rhs is overwritten.
            void Solve(const ValueType* the_A_matrix,
                       OrdinalType      the_A_matrix_extent,
                       OrdinalType      the_M_matrix_extent,
                       ValueType*       the_rhs) const;

        protected:
            ExecutionSpaceType the_execution_space_;
        };


        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRSV methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        BatchedBlockTridiagonalByBlockTRSV<BatchTraits,
                                           kSmallBlockExtent>::BatchedBlockTridiagonalByBlockTRSV(ExecutionSpaceType an_execution_space)
            : the_execution_space_{an_execution_space} {
            // EMPTY
        }

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        void BatchedBlockTridiagonalByBlockTRSV<BatchTraits,
                                                kSmallBlockExtent>::Solve(const ValueType* the_A_matrix,
                                                                          OrdinalType      the_A_matrix_extent,
                                                                          OrdinalType      the_M_matrix_extent,
                                                                          ValueType*       the_rhs) const {

            KOLIOP_ASSERT(the_A_matrix != nullptr);
            KOLIOP_ASSERT(the_rhs != nullptr);
            // the_A_matrix_extent must be divisible by
            // BatchTraitsType::kBatchSize. Else we need padding.
            KOLIOP_ASSERT(the_A_matrix_extent % BatchTraitsType::kBatchSize == 0);
            KOLIOP_ASSERT(the_M_matrix_extent > 0);

            const OrdinalType kBatchCount     = BatchTraitsType::BatchCount(the_A_matrix_extent);
            const OrdinalType kBigBlockExtent = the_M_matrix_extent;

            static_assert(MatrixKindType::kDiagonalCount == 3, "");

            typename MatrixKindType::ConstVectorType the_A_matrix_view{reinterpret_cast<const typename BatchTraitsType::ThreadVectorType*>(the_A_matrix),
                                                                       static_cast<std::size_t>(kBatchCount),
                                                                       static_cast<std::size_t>(kBigBlockExtent)};

            typename VectorKindType::VectorType the_rhs_vector_view{reinterpret_cast<typename BatchTraitsType::ThreadVectorType*>(the_rhs),
                                                                    static_cast<std::size_t>(kBatchCount),
                                                                    static_cast<std::size_t>(kBigBlockExtent)};

            const auto the_kernel = KOKKOS_LAMBDA(auto the_batch_element_index, auto the_batch_index) {

                auto the_matrix_batch_element = Kokkos::subview(the_A_matrix_view,
                                                                the_batch_index,
                                                                // TODO(Etienne M): These ALL() are constant, should we
                                                                // hardcode them to facilitate the compiler's work ?
                                                                Kokkos::ALL(), // kBigBlockExtent
                                                                Kokkos::ALL(), // kDiagonalCount
                                                                Kokkos::ALL(), // kSmallBlockExtent
                                                                Kokkos::ALL(), // kSmallBlockExtent
                                                                the_batch_element_index);

                auto the_vector_batch_element = Kokkos::subview(the_rhs_vector_view,
                                                                the_batch_index,
                                                                // TODO(Etienne M): These ALL() are constant, should we
                                                                // hardcode them to facilitate the compiler's work ?
                                                                Kokkos::ALL(), // kBigBlockExtent
                                                                Kokkos::ALL(), // kSmallBlockExtent
                                                                the_batch_element_index);

                // M =
                // B_0 C_0       0
                // A_1 B_1 C_1
                //     A_. A_. B_.
                // 0       A_n B_n
            };

            const auto a_policy = Kokkos::MDRangePolicy<ExecutionSpaceType,
                                                        Kokkos::Rank<2,
                                                                     // TODO(Etienne M): Tune blocking.
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{the_execution_space_,
                                                                                             {0, 0},
                                                                                             {BatchTraitsType::kVectorPerBatch, kBatchCount}};

            Kokkos::parallel_for("BatchedBlockTridiagonalByBlockTRSV::Solve",
                                 a_policy, the_kernel);
        }

    } // namespace blas
} // namespace koliop

#endif
