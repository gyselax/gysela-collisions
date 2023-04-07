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
        ///
        /// Assuming we have a block tridiagonal matrix:
        ///
        /// M =
        /// A_0 B_0       0
        /// C_0 A_1 B_1
        ///     C_1 A_. B_.
        /// 0       C_. A_n
        ///
        /// where A, B and C are small matrices instead of scalars. These small
        /// matrices are, in the case of Gysela, 3x3. Except for A, B and C
        /// everything is assumed zero.
        ///
        /// Then, we propose a special case LU to solve for this kind of
        /// matrices:
        ///
        /// M_0       0
        ///     M_.
        /// 0       M_n
        ///
        /// Due to the independent nature of the M_n blocks, we can treat them
        /// in batch. In practice, this means that:
        /// - On a CPU, each SIMD lane handles the factorization and solving for
        /// an M block. A thead is given a set of M contiguous blocks.
        /// - On a GPU, each thread    handles the factorization and solving for
        /// an M block. A team (thread block/workgroup) is given a set of M
        /// contiguous blocks.
        ///
        /// NOTE:
        /// - We do not use pivoting which, for numerical stability or simply
        /// validity of the solution, requires the A, B and C blocks be
        /// diagonally dominant.
        /// - We use a recursive LU decomposition very much similar to what is
        /// used in Linpack implementations. A description is given in
        /// Introduction to Algorithms, ISBN 978-0-262-03293-3 by Cormen, Thomas
        /// H. et al., page 819.
        ///
        class LUBlockTridiagonalByBlock {
        public:
            template <typename Array5D>
            static void Factorize(Array5D&);
            template <typename Array5D>
            static void Solve();

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
