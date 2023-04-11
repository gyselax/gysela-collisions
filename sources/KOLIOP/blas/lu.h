// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_LU_H
#define KOLIOP_BLAS_LU_H

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_LU_Team_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Team_Impl.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // LUBlockTridiagonalByBlock definition
        ////////////////////////////////////////////////////////////////////////

        /// Lower upper (LU) tooling for block tridiagonal by block matrices.
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
        /// We have 0 <= n < A_width. A_width is the number of independent M
        /// block of size kBigBlockExtent x kBigBlockExtent.
        ///
        /// The A matrix is expected to be stored under the following format:
        ///     View<Scalar *ccccc>{A_width / kBatchSize, kBigBlockExtent, 3, kSmallBlockExtent, kSmallBlockExtent, kBatchSize}
        /// The matrix should be in layout right and we justify that due to a
        /// limitation in Kokkos for representing constant extent, indeed they
        /// can not appear before dynamic extent.
        ///
        /// The more naive format below prevent vectorization on CPU and
        /// coalesced memory operations on GPU:
        ///     View<Scalar *cccc>{A_width              , kBigBlockExtent, 3, kSmallBlockExtent, kSmallBlockExtent}
        ///
        /// NOTE:
        /// In Gysela, kBigBlockExtent = Npar, kSmallBlockExtent = Npolmax and
        /// A_width depends on the blocking realized on r, theta and phi.
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
        ///
        template <typename Float,
                  typename ExecutionSpace>
        class LUBlockTridiagonalByBlock {
        public:
            using FloatType          = Float;
            using ExecutionSpaceType = ExecutionSpace;
            // TODO(Etienne M): Batch size

        public:
            LUBlockTridiagonalByBlock(ExecutionSpace an_execution_space);

            /// @brief TRF
            /// @param the_A_matrix Must respect the format described in the
            /// LUBlockTridiagonalByBlock notes.
            template <typename Array6D>
            void Factorize(Array6D&& the_A_matrix) const;

            /// @brief TRSV
            /// @param the_A_matrix Must respect the format described in the
            /// LUBlockTridiagonalByBlock notes.
            /// @param the_rhs TODO(Etienne M)
            template <typename Array6D,
                      typename Array1D>
            void SolveVector(const Array6D& the_A_matrix,
                             Array1D&&      the_rhs) const;

        protected:
            ExecutionSpace the_execution_space_;
        };


        ////////////////////////////////////////////////////////////////////////
        // LUBlockTridiagonalByBlock methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename ExecutionSpace>
        LUBlockTridiagonalByBlock<ExecutionSpace>::LUBlockTridiagonalByBlock(ExecutionSpace an_execution_space)
            : the_execution_space_{an_execution_space} {
            // EMPTY
        }

        template <typename ExecutionSpace>
        template <typename Array6D>
        void LUBlockTridiagonalByBlock<ExecutionSpace>::Factorize(Array6D&& the_A_matrix) const {
            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle){
                
            };

            // TODO(Etienne M): Assert that the range divide evenly by vectors
            // and other *grandeur*.

            // 1: We wish to saturate the hardware.
            // - On CPU this means using SIMD instructions.
            // - On GPU this means using (intra lane) SIMD such as double2 combined
            // with wavefront coalesced accesses. On AMD GPUs, a wavefront is 64
            // but we could go even lower and only seek to do 16 thread coalesced
            // accesses. Indeed 16 x sizeof(double) = 128 which is the cache lane
            // size.

            // kBatchSize represents the vector length from the hardware POV. The
            // vector length from the thread POV may differ due to intra lane SIMD
            // such as double2.
            static constexpr uint64_t kBatchSize  = DefaultVectorLength<FloatType, memory_space_type>::value;
            const uint64_t            kBatchCount = A_width / kBatchSize;

            Kokkos::View<FloatType******, Kokkos::LayoutRight, exec_space_type> A_storage("A",
                                                                                           kBatchCount,
                                                                                           kBigBlockExtent, 3,
                                                                                           kSmallBlockExtent, kSmallBlockExtent,
                                                                                           kBatchSize);

            // 2: Then we define the vector type from the thread POV.
            // - On CPU it is similar to the vector type from the hardware POV say,
            // a ymm register.
            // - On GPU it shall represent an inter lane SIMD such as double4.
            static constexpr uint64_t kVectorSize     = DefaultInternalVectorLength<FloatType, memory_space_type>::value;
            static constexpr uint64_t kVectorPerBatch = kBatchSize / kVectorSize;

            using ThreadVector = Vector<SIMD<FloatType>, kVectorSize>;

            Kokkos::View<ThreadVector******, Kokkos::LayoutRight, exec_space_type> A_vector(reinterpret_cast<ThreadVector*>(A_storage.data()),
                                                                                            // The batch count does not change.
                                                                                            kBatchCount,
                                                                                            kBigBlockExtent, 3,
                                                                                            kSmallBlockExtent, kSmallBlockExtent,
                                                                                            // But the batch potentially gets
                                                                                            // subdivided in vectors.
                                                                                            // NOTE: We changed the View's
                                                                                            // type.
                                                                                            kVectorPerBatch);

            // 3: Then, each thread identifies its batch and vector inside the
            // batch.

            const uint64_t kConcurrency   = XXX.concurency();
            const uint64_t kWorkItemCount = kBatchCount * kVectorPerBatch;
            const uint64_t kTeamSize      = WHAT here ? ; // 256; // 1 on CPU, at least a wavefront on GPUs ?
            const uint64_t kLeagueCount   = WHAT here ? ; // kWorkItemCount / kTeamSize;

            const uint64_t kVectorSize = 1; // We already defined ThreadVector so this vector loop is unnecessary.

            const auto a_policy = Kokkos::TeamPolicy<ExecutionSpace>(kLeagueCount, Kokkos::AUTO, kVectorSize);

            Kokkos::parallel_for("LUBlockTridiagonalByBlock::Factorize",
                                 a_policy, the_kernel);
        }

    } // namespace blas
} // namespace koliop

#endif
