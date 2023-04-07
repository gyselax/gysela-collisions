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
        /// where A, B and C are small matrices of size kSmallBlockWidth x
        /// kSmallBlockWidth instead of scalars. Except for A, B and C
        /// everything is assumed zero. We have 0 <= n < kBigBlockWidth.
        /// kBigBlockWidth is the number of block of size kSmallBlockWidth x
        /// kSmallBlockWidth.
        ///
        /// We propose LU tooling for this kind of matrix:
        ///
        /// A =
        /// M_0       0
        ///     M_.
        /// 0       M_n
        ///
        /// We have 0 <= n < A_width. A_width is the number of independent M
        /// blocks of size kBigBlockWidth x kBigBlockWidth.
        ///
        /// The A matrix is expected to be stored under the following format:
        ///     View<Scalar ccccc*>{kVectorWidth, kSmallBlockWidth, kSmallBlockWidth, 3, kBigBlockWidth, A_width / kVectorWidth}
        /// The more naive format below prevent vectorization on CPU and
        /// coalesced memory operations on GPU:
        ///     View<Scalar cccc* >{              kSmallBlockWidth, kSmallBlockWidth, 3, kBigBlockWidth, A_width}
        ///
        ///
        /// NOTE:
        /// In Gysela, kBigBlockWidth = Npar, kSmallBlockWidth = Npolmax and
        /// A_width depends on the blocking realized on r, theta and phi.
        ///
        /// NOTE:
        /// - We do not use pivoting which, for numerical stability or simply
        /// validity (division by 0) of the solution, requires in practice that
        /// the A, B and C blocks be diagonally dominant.
        /// - We use a recursive LU decomposition very much similar to what is
        /// used in Linpack implementations. A description is given in
        /// Introduction to Algorithms, ISBN 978-0-262-03293-3 by Cormen, Thomas
        /// H. et al., page 819.
        /// - Due to the independent nature of the M_n blocks, we can treat them
        /// in batch. In practice, this means that:
        ///     - On a CPU, each SIMD lane handles the factorization and solving
        ///       for an M block. A thead is given a set of M contiguous blocks.
        ///     - On a GPU, each thread    handles the factorization and solving
        ///       for an M block. A team (thread block/workgroup) is given a set
        ///       of M contiguous blocks.
        ///
        class LUBlockTridiagonalByBlock {
        public:
            /// @brief TRF
            /// @param the_A_matrix Must respect the format described in the
            /// LUBlockTridiagonalByBlock notes.
            template <typename Array6D>
            static void
            Factorize(Array6D&& the_A_matrix);

            /// @brief TRSV
            /// @param the_A_matrix Must respect the format described in the
            /// LUBlockTridiagonalByBlock notes.
            /// @param the_rhs TODO(Etienne M)
            template <typename Array6D,
                      typename Array1D>
            static void
            SolveVector(const Array6D& the_A_matrix,
                        Array1D&&      the_rhs);

        protected:
            static void DoFactorization() {
                // Inputs Matrix A with D diagonals by block with block of size
                // W. A is N block wide in width and height. A(i,j) denotes the
                // block at row i and column j.

                // assert(D >= 1);
                // assert(D % 2 == 1); // D is odd.

                // for(pivot = 0; pivot < (N - /* Last row */ 1); ++pivot) {
                //     pivot_row   = p;
                //     current_row = p + 1;

                //     if(D == 1) {
                //         A(p, p) = LUDecompose(A(p, p));
                //     } else {
                //         // NOTE: This can be implemented like so:
                //         // tmp = LU(A(p, p))
                //         // # tmp is now equal to L+U-I
                //         // # Then solve L*(U*X)=B with B = I
                //         // tmp2 = TRSM(tmp,  I, L) # Or Trtri
                //         // tmp  = TRSM(tmp2, I, U) # Or Trtri
                //         // # Then do A(current_row, p) * A(p, p)^-1
                //         // GEMM(A(current_row, p), tmp)
                //         A(current_row, p) = A(current_row, p) / A(p, p);

                //         // NOTE: If the matrix is tridiagonal, the following
                //         // loop can obviously be unrolled.

                //         // Do not overflow the matrix when near its bottom left.
                //         last_diagonal_column = min(pivot + D - 1, N);

                //         for(diag = pivot + 1; diag < last_diagonal_column; ++diag) {
                //             // NOTE: This can be implemented with a GEMM.
                //             A(current_row, diag) = A(current_row, diag) - A(pivot_row, diag) * A(current_row, p)
                //         }
                //     }
                // }
            }
        };


        ////////////////////////////////////////////////////////////////////////
        // LUBlockTridiagonalByBlock methods definition
        ////////////////////////////////////////////////////////////////////////


    } // namespace blas
} // namespace koliop

#endif
