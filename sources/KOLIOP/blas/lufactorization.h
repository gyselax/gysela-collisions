// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_LUFACTORIZATION_H
#define KOLIOP_BLAS_LUFACTORIZATION_H

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
        // LUFactorization definition
        ////////////////////////////////////////////////////////////////////////

        ///
        ///
        template <typename Float>
        class LUFactorization {
            public:
            public:
                void operator();

            protected:
            void DoFactorization() {
                // // General scalar LU decomp ~
                // const SizeType the_matrix_width = the_input.ColumnCount();

                // for(SizeType the_pivot_idx = 0; the_pivot_idx < the_matrix_width; ++the_pivot_idx) {

                //     const SizeType the_input_row_offset = the_pivot_idx + 1;
                //     const T        the_pivot_value_inv  = static_cast<T>(1.0F) / the_input.Scalar(the_pivot_idx, the_pivot_idx);

                //     for(SizeType the_input_y = the_input_row_offset; the_input_y < the_matrix_width; ++the_input_y) {

                //         // the_input.Scalar(:, the_pivot_idx) = scale(the_input.Scalar(:, the_pivot_idx), the_pivot_value_inv)
                //         const T the_row_scale_value                  = the_input.Scalar(the_input_y, the_pivot_idx) * the_pivot_value_inv;
                //         the_input.Scalar(the_input_y, the_pivot_idx) = the_row_scale_value;

                //         for(SizeType the_input_x = the_input_row_offset; the_input_x < the_matrix_width; ++the_input_x) {
                //             // the_input.Scalar(the_input_y, :) = saxpy(the_input.Scalar(the_input_y, :), - /* minus */ the_row_scale_value, the_input.Scalar(the_pivot_idx, the_input_x))
                //             the_input.Scalar(the_input_y, the_input_x) -= the_row_scale_value *
                //                                                           the_input.Scalar(the_pivot_idx, the_input_x);
                //         }
                //     }
                // }

                // // Inputs Matrix A with D diagonals by block with block of size
                // // W. A is N block wide in width and height. A(i,j) denotes the
                // // block at row i and column j.

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
        // LUFactorization methods definition
        ////////////////////////////////////////////////////////////////////////


    } // namespace blas
} // namespace koliop

#endif
