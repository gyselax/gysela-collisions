// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_TRSV_H
#define KOLIOP_BLAS_TRSV_H

#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>
#include <KokkosBlas2_gemv.hpp>

#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRSV definition
        ////////////////////////////////////////////////////////////////////////

        /// See BatchedBlockTridiagonalByBlockTRSV for a detailed matrix
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
            // using ConcurrencyPolicy = KokkosBatched::Mode::Serial;
            using CachingPolicy = KokkosBatched::Algo::Level3::Unblocked;

            using TRSVLower = KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Lower,
                                                        KokkosBatched::Trans::NoTranspose,
                                                        KokkosBatched::Diag::Unit,
                                                        CachingPolicy>;

            using TRSVUpper = KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Upper,
                                                        KokkosBatched::Trans::NoTranspose,
                                                        KokkosBatched::Diag::NonUnit,
                                                        CachingPolicy>;

            using GEMV = KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
                                                   KokkosBatched::Trans::NoTranspose,
                                                   CachingPolicy>;

        public:
            BatchedBlockTridiagonalByBlockTRSV(ExecutionSpaceType an_execution_space);

            /// TRSV, solve Ax = b where A is a LU factorization output
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

                // Solves Mx = vec, overwriting vec with x.

                // M =
                // B_0 C_0       0
                // A_1 B_1 C_1
                //     A_. A_. B_.
                // 0       A_n B_n

                // the_rhs =
                // b_0
                // b_1
                // b_.
                // b_n

                // B_n = L x U
                // With kSmallBlockExtent == 3
                // L =
                // 1   0
                // a 1
                // b c 1

                // U =
                // a b c
                //   d e
                // 0   f

                // b_n =
                // a
                // b
                // c

                // Forward substitution
                {
                    // Applying the naming above, we have:
                    auto B      = Kokkos::subview(the_matrix_batch_element, 0, 1, Kokkos::ALL(), Kokkos::ALL());
                    auto A_next = Kokkos::subview(the_matrix_batch_element, 0 /* Putting 1 here is illegal ? */, 0, Kokkos::ALL(), Kokkos::ALL());

                    auto b      = Kokkos::subview(the_vector_batch_element, 0, Kokkos::ALL());
                    auto b_next = Kokkos::subview(the_vector_batch_element, 0 /* Putting 1 here is illegal ? */, Kokkos::ALL());

                    for(OrdinalType the_small_block_index = 0;
                        the_small_block_index != (kBigBlockExtent - 1);
                        ++the_small_block_index) {

                        // TODO(Etienne M): Recreate view or assign ? Which is
                        // faster ?

                        B.assign_data(&the_matrix_batch_element(the_small_block_index, 1, 0, 0));

                        TRSVLower::invoke(1.0, B, b); // Solve: B_lower * y = b, y is put into b

                        A_next.assign_data(&the_matrix_batch_element(the_small_block_index + 1, 0, 0, 0));
                        b_next.assign_data(&the_vector_batch_element(the_small_block_index + 1, 0));

                        GEMV::invoke(-1.0, A_next, b, 1.0, b_next); // b_next = 1.0 * b_next + -1.0 * A_next * b

                        b = b_next;
                    }

                    B.assign_data(&the_matrix_batch_element(kBigBlockExtent - 1, 1, 0, 0));

                    TRSVLower::invoke(1.0, B, b);
                }

                // Backward substitution
                {
                    // Applying the naming above, we have:
                    auto B      = Kokkos::subview(the_matrix_batch_element, 0, 1, Kokkos::ALL(), Kokkos::ALL());
                    auto C_next = Kokkos::subview(the_matrix_batch_element, 0, 0, Kokkos::ALL(), Kokkos::ALL());

                    auto b      = Kokkos::subview(the_vector_batch_element, kBigBlockExtent - 1, Kokkos::ALL());
                    auto b_next = Kokkos::subview(the_vector_batch_element, 0, Kokkos::ALL());

                    // NOTE: *next* when iterating in reverse order is
                    // *previous* when iterating in natural order.

                    for(OrdinalType the_reversed_small_block_index = kBigBlockExtent;
                        the_reversed_small_block_index != 1;
                        --the_reversed_small_block_index) {

                        const OrdinalType the_small_block_index = the_reversed_small_block_index - 1;

                        // TODO(Etienne M): Recreate view or assign ? Which is
                        // faster ?

                        B.assign_data(&the_matrix_batch_element(the_small_block_index, 1, 0, 0));

                        TRSVUpper::invoke(1.0, B, b); // Solve: B_upper * x = b, x is put into b

                        C_next.assign_data(&the_matrix_batch_element(the_small_block_index - 1, 2, 0, 0));
                        b_next.assign_data(&the_vector_batch_element(the_small_block_index - 1, 0));

                        GEMV::invoke(-1.0, C_next, b, 1.0, b_next); // b_next = 1.0 * b_next + -1.0 * C_next * b

                        b = b_next;
                    }

                    B.assign_data(&the_matrix_batch_element(0, 1, 0, 0));

                    TRSVLower::invoke(1.0, B, b);
                }
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
