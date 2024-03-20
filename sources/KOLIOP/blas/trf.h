// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_TRF_H
#define KOLIOP_BLAS_TRF_H

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>

#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRF definition
        ////////////////////////////////////////////////////////////////////////

        /// Lower Upper (LU) tooling for block tridiagonal by block matrices.
        /// We treat square matrices.
        ///
        /// Assuming we have a block tridiagonal matrix:
        ///
        /// M =
        /// B_0 C_0       0
        /// A_1 B_1 C_1
        ///     A_. B_. C_.
        /// 0       A_n B_n
        ///
        /// where A, B and C are small matrices of size kSmallBlockExtent x
        /// kSmallBlockExtent instead of scalars. Except for A, B and C
        /// everything is assumed zero. We have 0 <= n < kBigBlockExtent.
        /// kBigBlockExtent is the number of block of size kSmallBlockExtent x
        /// kSmallBlockExtent.
        /// A_0 and C_n are expected to be zero.
        ///
        /// We propose LU tooling for this kind of matrix:
        ///
        /// A =
        /// M_0       0
        ///     M_.
        /// 0       M_n
        ///
        /// We have 0 <= n < A_matrix_extent. A_matrix_extent is the number of
        /// independent M block of size kBigBlockExtent x kBigBlockExtent.
        ///
        /// The A matrix is expected to be stored under the following format:
        ///     View<Scalar **cccc>{A_matrix_extent / BatchTraitsType::kBatchSize, kBigBlockExtent, kDiagonalCount, kSmallBlockExtent, kSmallBlockExtent, BatchTraitsType::kBatchSize}
        /// The first kSmallBlockExtent represents the number of row and the
        /// second represents the columns so that:
        /// ScalarMatrixType{}(5, 4, 2, 2, 1, 0)
        /// represents the first (0) batch element of the second (1) column of
        /// the third (2) row of the upper (2) diagonal block at the fifth row
        /// (4) of the sixth batch (5).
        ///
        /// The matrix should be in layout right and we justify that due to a
        /// limitation in Kokkos for representing constant extent, indeed they
        /// can not appear before dynamic extent. The A is followed by the block
        /// B and C like so:
        /// [ A_matrix_extent / BatchTraitsType::kBatchSize
        ///     [ kBigBlockExtent
        ///         [ kDiagonalCount
        ///             [BlockA], [BlockB], [BlockC]
        ///         ]
        ///     ]
        /// ]
        /// Each block contains the data for potentially multiple (BatchTraitsType::kBatchSize)
        /// problems like so:
        /// [ BlockA
        ///     [ BlockA row values
        ///         [ BatchTraitsType::kBatchSize values]
        ///     ]
        /// ]
        ///
        /// For the_rhs vector we expect the following format:
        ///     View<value_type **cc>{A_matrix_extent / BatchTraitsType::kBatchSize, kBigBlockExtent, kSmallBlockExtent, BatchTraitsType::kBatchSize}
        ///
        /// NOTE:
        /// In Gysela, kBigBlockExtent = Nvpar, kSmallBlockExtent = Npolmax and
        /// A_matrix_extent depends on the blocking realized on r, theta and
        /// phi.
        ///
        /// NOTE:
        /// - We do not use pivoting which, for numerical stability or simply
        /// validity (division by 0) of the solution, requires in practice that
        /// the M blocks be diagonally dominant.
        /// - It is the user's responsibility to wait (via fence or else) for
        /// the operation's completion.
        /// - We use a recursive LU decomposition very much similar to what is
        /// used in Linpack implementations. A description is given in
        /// Introduction to Algorithms, ISBN 978-0-262-03293-3 by Cormen, Thomas
        /// H. et al., page 819.
        /// That is, we observe that:
        /// A =      L =      U =
        /// A11 A12  L11 0    U11 U12
        /// A21 A22  L21 L22  0   U22 where A = L U.
        /// We thus have:
        /// A11 = L11 * U11             # LU decomposition of a small block.
        /// A12 = L11 * U12             # TRSM, solve for U12.
        /// A21 = L21 * U11             # TRSM, solve for L21.
        /// A22 = L21 * U12 + L22 * U22 # GEMM, solve for L22 * U22 which
        ///                                     becomes the new A22.
        /// Then recursively work on factorizing A22 = L22 * U22.
        ///
        /// So factorizing A can be done by doing a smaller LU factorization of
        /// A11 and then, doing two TRSM and a GEMM.
        ///
        /// - Due to the independent nature of the M_n blocks, we can treat them
        /// in batch of BatchTraitsType::kBatchSize per thread or team. In
        /// practice, this means that:
        ///     - On a CPU, each SIMD lane handles the factorization and solving
        ///       for an M block. A thead is given a set of M contiguous blocks.
        ///     - On a GPU, each thread    handles the factorization and solving
        ///       for an M block. A team (thread block/workgroup) is given a set
        ///       of M contiguous blocks. On GPU we may also use SIMD in the
        ///       form of double2 or float2 etc. (SIMDception).
        ///     - An M block is referred as batch element.
        ///     - The number of M block must be a multiple of
        ///       BatchTraitsType::kBatchSize (you can add padding).
        ///
        template <typename BatchTraits,
                  // TODO(Etienne M): See if kFlattenedBlockExtent which is
                  // constexpr, can help improve the performance when used as an
                  // template argument. kBigBlockExtent = kFlattenedBlockExtent.
                  // This would remove the_A_matrix_extent argument.
                  OrdinalType kSmallBlockExtent>
        class BatchedBlockTridiagonalByBlockTRF {
        public:
            using BatchTraitsType = BatchTraits;

            using value_type = typename BatchTraitsType::value_type;

            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

            using MatrixKindType = kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                kSmallBlockExtent>;

        protected:
            // using ConcurrencyPolicy = KokkosBatched::Mode::Serial;

            // TODO(Etienne M): Compare Unblocked vs blocked on GPU and CPUs.
            using CachingPolicy = KokkosBatched::Algo::Level3::Unblocked;

            using LU = KokkosBatched::SerialLU<CachingPolicy>;

            using TRSMLower = KokkosBatched::SerialTrsm<KokkosBatched::Side::Left,
                                                        KokkosBatched::Uplo::Lower,
                                                        KokkosBatched::Trans::NoTranspose,
                                                        KokkosBatched::Diag::Unit,
                                                        CachingPolicy>;

            using TRSMUpper = KokkosBatched::SerialTrsm<KokkosBatched::Side::Right,
                                                        KokkosBatched::Uplo::Upper,
                                                        KokkosBatched::Trans::NoTranspose,
                                                        KokkosBatched::Diag::NonUnit,
                                                        CachingPolicy>;

            using GEMM = KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
                                                   KokkosBatched::Trans::NoTranspose,
                                                   CachingPolicy>;

        public:
            /// TRF
            /// @param the_A_matrix Must respect the format described in the
            /// BatchedBlockTridiagonalByBlockTRF notes.
            /// @param the_A_matrix_extent The number of independent block
            /// diagonal problem. Must be a multiple of
            /// BatchTraitsType::kBatchSize.
            /// @param the_M_matrix_extent The number extent of the block
            /// diagonal problem.
            void Factorize(ExecutionSpaceType an_execution_space,
                           value_type*        the_A_matrix,
                           OrdinalType        the_A_matrix_extent,
                           OrdinalType        the_M_matrix_extent) const;

        protected:
        };


        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRF methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        void BatchedBlockTridiagonalByBlockTRF<BatchTraits,
                                               kSmallBlockExtent>::Factorize(ExecutionSpaceType an_execution_space,
                                                                             value_type*        the_A_matrix,
                                                                             OrdinalType        the_A_matrix_extent,
                                                                             OrdinalType        the_M_matrix_extent) const {

            KOLIOP_ASSERT(the_A_matrix != nullptr);
            // the_A_matrix_extent must be divisible by
            // BatchTraitsType::kBatchSize. Else we need padding.
            KOLIOP_ASSERT(the_A_matrix_extent % BatchTraitsType::kBatchSize == 0);
            KOLIOP_ASSERT(the_M_matrix_extent > 0);

            const OrdinalType kBatchCount     = BatchTraitsType::BatchCount(the_A_matrix_extent);
            const OrdinalType kBigBlockExtent = the_M_matrix_extent;

            static_assert(MatrixKindType::kDiagonalCount == 3, "");

            typename MatrixKindType::VectorType the_A_matrix_view{reinterpret_cast<typename BatchTraitsType::ThreadVectorType*>(the_A_matrix),
                                                                  static_cast<std::size_t>(kBatchCount),
                                                                  static_cast<std::size_t>(kBigBlockExtent)};

            // TODO(Etienne M): Check that we do not benefit from using
            // TeamLU/TRSM/GEMM. This require a trick with vector threads.

            // NOTE:
            // Due to memory consumption issues (storing the whole system), we
            // have to do blocking. Due to the blocking we may not be able to
            // utilize the GPU enough. Say 64 t/wavefront * 32 wavefront/cu *
            // 110 cu for a total of 225280 potential in flight workitem per
            // MI250 GCD. This means, assuming 1024 nvpar and Flo64:
            // (8 * 1024 * (3 * (3*3))) * 225280 = 49'828'331'520 o.

            const auto the_kernel = KOKKOS_LAMBDA(OrdinalType the_batch_element_index, OrdinalType the_batch_index) {

                auto the_batch_element = Kokkos::subview(the_A_matrix_view,
                                                         the_batch_index,
                                                         // TODO(Etienne M): These ALL() are constant, should we
                                                         // hardcode them to facilitate the compiler's work ?
                                                         Kokkos::ALL(), // kBigBlockExtent
                                                         Kokkos::ALL(), // kDiagonalCount
                                                         Kokkos::ALL(), // kSmallBlockExtent
                                                         Kokkos::ALL(), // kSmallBlockExtent
                                                         the_batch_element_index);

                // M =
                // B_0 C_0       0
                // A_1 B_1 C_1
                //     A_. A_. B_.
                // 0       A_n B_n

                // Applying the naming above, we have:
                auto B      = Kokkos::subview(the_batch_element, 0, 1, Kokkos::ALL(), Kokkos::ALL());
                auto C      = Kokkos::subview(the_batch_element, 0, 2, Kokkos::ALL(), Kokkos::ALL());
                auto A_next = Kokkos::subview(the_batch_element, 0, 0, Kokkos::ALL(), Kokkos::ALL());
                auto B_next = Kokkos::subview(the_batch_element, kBigBlockExtent - 1, 1, Kokkos::ALL(), Kokkos::ALL());

                for(OrdinalType the_small_block_index = 0;
                    the_small_block_index < (kBigBlockExtent - 1);
                    ++the_small_block_index) {

                    // TODO(Etienne M): Recreate view or assign ? Which is
                    // faster ?
                    // NOTE: Is using restricted view correct here ? While none
                    // of the B, C, A_next and B_next alias in a given iteration
                    // if there is unrolling then B_next becomes B.

                    B.assign_data(&the_batch_element(the_small_block_index, 1, 0, 0));
                    C.assign_data(&the_batch_element(the_small_block_index, 2, 0, 0));
                    A_next.assign_data(&the_batch_element(the_small_block_index + 1, 0, 0, 0));
                    B_next.assign_data(&the_batch_element(the_small_block_index + 1, 1, 0, 0));

                    LU::invoke(B);                                              // B = L + U - I
                    TRSMLower::invoke(static_cast<value_type>(1.0), B, C);      // Solve: B_lower * X = C, X is put into C
                    TRSMUpper::invoke(static_cast<value_type>(1.0), B, A_next); // Solve: B_upper * X = A_next, X is put into A_next
                    GEMM::invoke(static_cast<value_type>(-1.0), A_next, C,      // B_next = 1.0 * B_next + -1.0 * A_next * C
                                 static_cast<value_type>(1.0), B_next);
                }

                LU::invoke(B_next);
            };

            const auto a_policy = Kokkos::MDRangePolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                ExecutionSpaceType,
                Kokkos::Rank<2,
                             // TODO(Etienne M): Tune blocking.
                             Kokkos::Iterate::Left,
                             Kokkos::Iterate::Left>>{an_execution_space,
                                                     {0, 0},
                                                     {BatchTraitsType::kVectorPerBatch, kBatchCount}};

            Kokkos::parallel_for("BatchedBlockTridiagonalByBlockTRF::Factorize",
                                 a_policy, the_kernel);
        }

    } // namespace blas
} // namespace koliop

#endif
