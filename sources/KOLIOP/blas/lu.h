// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_LU_H
#define KOLIOP_BLAS_LU_H

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // LUBlockTridiagonalByBlock definition
        ////////////////////////////////////////////////////////////////////////

        /// Lower Upper (LU) tooling for block tridiagonal by block matrices.
        /// We treat square matrices.
        ///
        /// Assuming we have a block tridiagonal matrix:
        ///
        /// M =
        /// A_0 B_0       0
        /// C_0 A_1 B_1
        ///     C_1 A_. B_.
        /// 0       C_. A_n
        ///
        /// where A, B and C are small matrices of size kSmallBlockExtent x
        /// kSmallBlockExtent instead of scalars. Except for A, B and C
        /// everything is assumed zero. We have 0 <= n < kBigBlockExtent.
        /// kBigBlockExtent is the number of block of size kSmallBlockExtent x
        /// kSmallBlockExtent.
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
        ///     View<Scalar **cccc>{A_matrix_extent / kBatchSize, kBigBlockExtent, kDiagonalCount, kSmallBlockExtent, kSmallBlockExtent, kBatchSize}
        /// The first kSmallBlockExtent represents the number of row and the
        /// second represents the columns so that:
        /// ScalarMatrixType{}(5, 4, 2, 2, 1, 0)
        /// represents the first (0) batch element of the second (1) column of
        /// the third (2) row of the upper (2) diagonal block at the fifth row
        /// (4) of batch sixth batch.
        ///
        /// The matrix should be in layout right and we justify that due to a
        /// limitation in Kokkos for representing constant extent, indeed they
        /// can not appear before dynamic extent. The A is followed by the block
        /// B and C like so:
        /// [ A_matrix_extent / kBatchSize
        ///     [ kBigBlockExtent
        ///         [ kDiagonalCount
        ///             [BlockA], [BlockB], [BlockC]
        ///         ]
        ///     ]
        /// ]
        /// Each block contains the data for potentially multiple (kBatchSize)
        /// problems like so:
        /// [ BlockA
        ///     [ BlockA row values
        ///         [ kBatchSize values]
        ///     ]
        /// ]
        ///
        /// For the_rhs vector we expect the following format:
        ///     View<FloatType **cc>{A_matrix_extent / kBatchSize, kBigBlockExtent, kSmallBlockExtent, kBatchSize}
        ///
        /// NOTE:
        /// In Gysela, kBigBlockExtent = Npar, kSmallBlockExtent = Npolmax and
        /// A_matrix_extent depends on the blocking realized on r, theta and
        /// phi.
        ///
        /// NOTE:
        /// - We do not use pivoting which, for numerical stability or simply
        /// validity (division by 0) of the solution, requires in practice that
        /// the A, B and C blocks be diagonally dominant.
        /// - It is the user's responsibility to wait (via fence or else) for
        /// the operation's completion.
        /// - We use a recursive LU decomposition very much similar to what is
        /// used in Linpack implementations. A description is given in
        /// Introduction to Algorithms, ISBN 978-0-262-03293-3 by Cormen, Thomas
        /// H. et al., page 819.
        /// - Due to the independent nature of the M_n blocks, we can treat them
        /// in batch of kBatchSize per thread or team. In practice, this means
        /// that:
        ///     - On a CPU, each SIMD lane handles the factorization and solving
        ///       for an M block. A thead is given a set of M contiguous blocks.
        ///     - On a GPU, each thread    handles the factorization and solving
        ///       for an M block. A team (thread block/workgroup) is given a set
        ///       of M contiguous blocks. On GPU we may also use SIMD in the
        ///       form of double2 or float2 etc. (SIMDception).
        ///     - An M block is referred as batch element.
        ///     - The a matrix must a number of M block a multiple of kBatchSize
        ///       (you can add padding).
        ///
        template <typename Float,
                  typename ExecutionSpace,
                  // TODO(Etienne M): See if kFlattenedBlockExtent which is
                  // constexpr, can help improve the performance when used as an
                  // template argument. kBigBlockExtent = kFlattenedBlockExtent.
                  OrdinalType kSmallBlockExtent = 3>
        class LUBlockTridiagonalByBlock {
        public:
            using FloatType          = Float;
            using ExecutionSpaceType = ExecutionSpace;

            /// We could make this more general but for now the code is assuming
            /// this is fixed to 3.
            static inline constexpr OrdinalType kDiagonalCount = 3;

            /// We wish to saturate the hardware.
            /// - On CPU this means using SIMD instructions.
            /// - On GPU this means using (intra lane) SIMD such as double2
            /// combined with wavefront coalesced accesses. On AMD GPUs, a
            /// wavefront is 64 but we could go even lower and only seek to do
            /// 16 thread coalesced accesses as wave fronts a processed by
            /// groups of 16 threads.
            /// kBatchSize represents the vector length from the hardware POV.
            /// The vector length from the thread POV may differ due to intra
            /// lane SIMD such as double2.
            static inline constexpr OrdinalType kBatchSize = KokkosBatched::DefaultVectorLength<FloatType,
                                                                                                typename ExecutionSpace::memory_space>::value;

            using ScalarMatrixType = Kokkos::View<FloatType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][kBatchSize],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpace>;

            using ScalarVectorType = Kokkos::View<FloatType** [kSmallBlockExtent][kBatchSize],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpace>;

            /// We define the vector type from the thread POV.
            /// - On CPU it is similar to the vector type from the hardware POV
            /// say, a ymm register.
            /// - On GPU it shall represent an inter lane SIMD such as double4.
            static inline constexpr OrdinalType kVectorSize = KokkosBatched::DefaultInternalVectorLength<FloatType,
                                                                                                         typename ExecutionSpace::memory_space>::value;

            static inline constexpr OrdinalType kVectorPerBatch = kBatchSize / kVectorSize;

            using ThreadVectorType = KokkosBatched::Vector<KokkosBatched::SIMD<FloatType>,
                                                           kVectorSize>;

            /// The batch count did not changed compared to ScalarMatrixType
            /// but the batch potentially gets subdivided in vectors.
            /// NOTE: We changed the View's type.
            using VectorMatrixType = Kokkos::View<ThreadVectorType** [kDiagonalCount][kSmallBlockExtent][kSmallBlockExtent][kVectorPerBatch],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpace>;

            using VectorVectorType = Kokkos::View<ThreadVectorType** [kSmallBlockExtent][kVectorPerBatch],
                                                  Kokkos::LayoutRight,
                                                  ExecutionSpace>;

        protected:
            // using ConcurrencyPolicy = KokkosBatched::Mode::Serial;
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
            LUBlockTridiagonalByBlock(ExecutionSpace an_execution_space);

            /// @brief TRF
            /// @param the_A_matrix Must respect the format described in the
            /// LUBlockTridiagonalByBlock notes.
            /// @param the_A_matrix_extent The number of independent block
            /// diagonal problem. Must be a multiple of kBatchSize.
            /// @param the_M_matrix_extent The number extent of the block
            /// diagonal problem.
            void Factorize(FloatType*  the_A_matrix,
                           OrdinalType the_A_matrix_extent,
                           OrdinalType the_M_matrix_extent) const;

            /// @brief TRSV
            /// @param the_A_matrix Must respect the format described in the
            /// LUBlockTridiagonalByBlock notes.
            /// @param the_A_matrix_extent The number of independent block
            /// diagonal problem. Must be a multiple of kBatchSize.
            /// @param the_M_matrix_extent The number extent of the block
            /// diagonal problem.
            /// @param the_rhs A vector of the_A_matrix_extent *
            /// the_M_matrix_extent * kSmallBlockExtent rows. for the format,
            /// see above.
            void SolveVector(const FloatType* the_A_matrix,
                             OrdinalType      the_A_matrix_extent,
                             OrdinalType      the_M_matrix_extent,
                             FloatType*       the_rhs) const;

        protected:
            ExecutionSpaceType the_execution_space_;
        };


        ////////////////////////////////////////////////////////////////////////
        // LUBlockTridiagonalByBlock methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Float,
                  typename ExecutionSpace,
                  OrdinalType kSmallBlockExtent>
        LUBlockTridiagonalByBlock<Float,
                                  ExecutionSpace,
                                  kSmallBlockExtent>::LUBlockTridiagonalByBlock(ExecutionSpace an_execution_space)
            : the_execution_space_{an_execution_space} {
            // EMPTY
        }

        template <typename Float,
                  typename ExecutionSpace,
                  OrdinalType kSmallBlockExtent>
        void LUBlockTridiagonalByBlock<Float,
                                       ExecutionSpace,
                                       kSmallBlockExtent>::Factorize(FloatType*  the_A_matrix,
                                                                     OrdinalType the_A_matrix_extent,
                                                                     OrdinalType the_M_matrix_extent) const {

            KOLIOP_ASSERT(the_A_matrix != nullptr);
            // the_A_matrix_extent must be divisible by kBatchSize. Else we need
            // padding.
            KOLIOP_ASSERT(the_A_matrix_extent % kBatchSize == 0);
            KOLIOP_ASSERT(the_M_matrix_extent > 0);

            const OrdinalType kBatchCount     = the_A_matrix_extent / kBatchSize;
            const OrdinalType kBigBlockExtent = the_M_matrix_extent;

            VectorMatrixType the_A_matrix_view{reinterpret_cast<ThreadVectorType*>(the_A_matrix),
                                               static_cast<std::size_t>(kBatchCount),
                                               static_cast<std::size_t>(the_M_matrix_extent)};

            // TODO(Etienne M): Check that we do not benefit from using
            // TeamLU/TRSM/GEMM. This require a trick with vector threads.

            // NOTE:
            // Due to memory consumption issues (storing the whole system), we
            // have to do blocking. Due to the blocking we may not be able to
            // utilize the GPU enough. Say 64 t/wavefront * 32 wavefront/cu *
            // 110 cu for a total of 225280 potential in flight workitem per
            // MI250 GCD. This means, assuming 1024 nvpar and Flo64:
            // (8 * 1024 * (3 * (3*3))) * 225280 = 49'828'331'520 o.

            const auto the_kernel = KOKKOS_LAMBDA(auto the_batch_index, auto the_batch_element_index) {
                static_assert(kDiagonalCount == 3, "");

                const auto the_batch_element = Kokkos::subview(the_A_matrix_view,
                                                               the_batch_index,
                                                               // TODO(Etienne M): These ALL() are constant, should we
                                                               // hardcode them to facilitate the compiler's work ?
                                                               Kokkos::ALL(), // kBigBlockExtent
                                                               Kokkos::ALL(), // kDiagonalCount
                                                               Kokkos::ALL(), // kSmallBlockExtent
                                                               Kokkos::ALL(), // kSmallBlockExtent
                                                               the_batch_element_index);

                // Applying the naming above, we have:
                auto A = Kokkos::subview(the_batch_element, 0, 0, Kokkos::ALL(), Kokkos::ALL());

                if(kBigBlockExtent == 1) {
                    LU::invoke(A);
                } else {
                    auto B      = Kokkos::subview(the_batch_element, 0, 1, Kokkos::ALL(), Kokkos::ALL());
                    auto C      = Kokkos::subview(the_batch_element, 0, 2, Kokkos::ALL(), Kokkos::ALL());
                    auto A_next = Kokkos::subview(the_batch_element, 1, 0, Kokkos::ALL(), Kokkos::ALL());

                    for(OrdinalType the_small_block_index = 0;
                        the_small_block_index < (kBigBlockExtent - 1);
                        ++the_small_block_index) {

                        // TODO(Etienne M): Recreate view or assign ? Which is
                        // faster ?

                        A.assign_data(&the_batch_element(the_small_block_index, 0, 0, 0));
                        B.assign_data(&the_batch_element(the_small_block_index, 1, 0, 0));
                        C.assign_data(&the_batch_element(the_small_block_index, 2, 0, 0));
                        A_next.assign_data(&the_batch_element(the_small_block_index + 1, 0, 0, 0));

                        LU::invoke(A);                         // A = L+U-I
                        TRSMLower::invoke(1.0, A, B);          // Solve: A_lower * X = B, X is put into B
                        TRSMUpper::invoke(1.0, A, C);          // Solve: A_upper * X = C, X is put into C
                        GEMM::invoke(-1.0, C, B, 1.0, A_next); // A_next = 1.0 * A_next + -1 * C * B
                    }

                    KokkosBatched::SerialLU<CachingPolicy>::invoke(A_next);
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<ExecutionSpaceType,
                                                        Kokkos::Rank<2,
                                                                     // TODO(Etienne M): Tune blocking.
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Right>>{the_execution_space_,
                                                                                              {0, 0},
                                                                                              {kBatchCount, kVectorPerBatch}};

            Kokkos::parallel_for("LUBlockTridiagonalByBlock::Factorize",
                                 a_policy, the_kernel);
        }

        template <typename Float,
                  typename ExecutionSpace,
                  OrdinalType kSmallBlockExtent>
        void LUBlockTridiagonalByBlock<Float,
                                       ExecutionSpace,
                                       kSmallBlockExtent>::SolveVector(const FloatType*,
                                                                       OrdinalType,
                                                                       OrdinalType,
                                                                       FloatType*) const {
        }

    } // namespace blas
} // namespace koliop

#endif
