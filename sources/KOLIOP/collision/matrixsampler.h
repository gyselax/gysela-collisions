// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_MATRIXSAMPLER_H
#define KOLIOP_COLLISION_MATRIXSAMPLER_H

#include <Kokkos_Macros.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // MatrixSampler definition
        ////////////////////////////////////////////////////////////////////////

        ///
        ///
        template <typename View6D>
        class MatrixSampler {
        public:
        public:
            MatrixSampler(const View6D&    the_Nia,
                          ComputeFloatType a_vpar_delta,
                          ComputeFloatType a_time_delta_coefficient);

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            LowerDiagonal(OrdinalType the_batch_index,
                          OrdinalType ivpar,
                          OrdinalType ipol,
                          OrdinalType jpol,
                          OrdinalType the_batch_element_index) const;

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            MiddleDiagonal(OrdinalType the_batch_index,
                           OrdinalType ivpar,
                           OrdinalType ipol,
                           OrdinalType jpol,
                           OrdinalType the_batch_element_index) const;

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            UpperDiagonal(OrdinalType the_batch_index,
                          OrdinalType ivpar,
                          OrdinalType ipol,
                          OrdinalType jpol,
                          OrdinalType the_batch_element_index) const;

            KOKKOS_INLINE_FUNCTION void
            AllDiagonals(OrdinalType       the_batch_index,
                         OrdinalType       ivpar,
                         OrdinalType       ipol,
                         OrdinalType       jpol,
                         OrdinalType       the_batch_element_index,
                         ComputeFloatType& the_lower_diagonal,
                         ComputeFloatType& the_middle_diagonal,
                         ComputeFloatType& the_upper_diagonal) const;

        protected:
            // TODO(Etienne M): having a copy may not be ideal. Maybe we should
            // pass everything the the sampling function.
            View6D           the_Nia_;
            ComputeFloatType the_time_delta_coefficient_;
            ComputeFloatType the_coefficients_[2];
        };


        ////////////////////////////////////////////////////////////////////////
        // MatrixSampler methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename View6D>
        MatrixSampler<View6D>::MatrixSampler(const View6D&    the_Nia,
                                             ComputeFloatType a_vpar_delta,
                                             ComputeFloatType a_time_delta_coefficient)
            : the_Nia_{the_Nia}
            , the_time_delta_coefficient_{a_time_delta_coefficient}
            , the_coefficients_{a_time_delta_coefficient / (a_vpar_delta * 2.0),
                                a_time_delta_coefficient / (a_vpar_delta * a_vpar_delta)} {
            // EMPTY
        }

        template <typename View6D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D>::LowerDiagonal(OrdinalType the_batch_index,
                                             OrdinalType ivpar,
                                             OrdinalType ipol,
                                             OrdinalType jpol,
                                             OrdinalType the_batch_element_index) const {
            // NOTE: Careful with the i,j. The value of the 3x3 matrix block are
            // given using transposed values of the Nia. With i the row and j
            // the column:
            // A[i, j] = -N1(j, i) * ...
            return the_Nia_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) * the_coefficients_[1] -
                   the_Nia_(the_batch_index, ivpar, 1, jpol, ipol, the_batch_element_index) * the_coefficients_[0];
        }

        template <typename View6D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D>::MiddleDiagonal(OrdinalType the_batch_index,
                                              OrdinalType ivpar,
                                              OrdinalType ipol,
                                              OrdinalType jpol,
                                              OrdinalType the_batch_element_index) const {
            const ComputeFloatType the_value = the_Nia_(the_batch_index, ivpar, 0, jpol, ipol, the_batch_element_index) * the_time_delta_coefficient_ -
                                               the_Nia_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) * the_coefficients_[1] * 2.0;


            // the_matrix = I - dt/4 * T

            // TODO(Etienne M): On CPU this branch should get moved up
            // in the loop and on GPU I hope it does not cost too much.
            // Else we have to launch another kernel on the diagonal.
            // TODO(Etienne M): What about branch divergence.

            return jpol == ipol ?
                       the_value + 1.0 :
                       the_value;
        }

        template <typename View6D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D>::UpperDiagonal(OrdinalType the_batch_index,
                                             OrdinalType ivpar,
                                             OrdinalType ipol,
                                             OrdinalType jpol,
                                             OrdinalType the_batch_element_index) const {
            return the_Nia_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) * the_coefficients_[1] +
                   the_Nia_(the_batch_index, ivpar, 1, jpol, ipol, the_batch_element_index) * the_coefficients_[0];
        }

        template <typename View6D>
        KOKKOS_INLINE_FUNCTION void
        MatrixSampler<View6D>::AllDiagonals(OrdinalType       the_batch_index,
                                            OrdinalType       ivpar,
                                            OrdinalType       ipol,
                                            OrdinalType       jpol,
                                            OrdinalType       the_batch_element_index,
                                            ComputeFloatType& the_lower_diagonal,
                                            ComputeFloatType& the_middle_diagonal,
                                            ComputeFloatType& the_upper_diagonal) const {
            the_lower_diagonal = LowerDiagonal(the_batch_index,
                                               ivpar,
                                               ipol,
                                               jpol,
                                               the_batch_element_index);

            the_middle_diagonal = MiddleDiagonal(the_batch_index,
                                                 ivpar,
                                                 ipol,
                                                 jpol,
                                                 the_batch_element_index);

            the_upper_diagonal = UpperDiagonal(the_batch_index,
                                               ivpar,
                                               ipol,
                                               jpol,
                                               the_batch_element_index);
        }
    } // namespace collision
} // namespace koliop

#endif
