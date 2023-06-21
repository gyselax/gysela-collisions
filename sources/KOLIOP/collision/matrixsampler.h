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
        template <typename View6D,
                  typename View1D>
        class MatrixSampler {
        public:
        public:
            MatrixSampler(const View6D&    the_Najl,
                          const View1D&    vparg,
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
            View6D           the_Najl_;
            View1D           vparg_;
            ComputeFloatType the_time_delta_coefficient_;
        };


        ////////////////////////////////////////////////////////////////////////
        // MatrixSampler methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename View6D,
                  typename View1D>
        MatrixSampler<View6D, View1D>::MatrixSampler(const View6D&    the_Najl,
                                                     const View1D&    vparg,
                                                     ComputeFloatType a_time_delta_coefficient)
            : the_Najl_{the_Najl}
            , vparg_{vparg}
            , the_time_delta_coefficient_{a_time_delta_coefficient} {
            // EMPTY
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::LowerDiagonal(OrdinalType the_batch_index,
                                                     OrdinalType ivpar,
                                                     OrdinalType ipol,
                                                     OrdinalType jpol,
                                                     OrdinalType the_batch_element_index) const {
            // NOTE: Careful with the i,j. The value of the 3x3 matrix block are
            // given using transposed values of the Najl. With i the row and j
            // the column:
            // A[i, j] = -N1(j, i) * ...
            // NOTE: vparg_(ivpar) - vparg_(ivpar + 1) would overflow
            const auto a_vpar_delta = vparg_(1) - vparg_(0);
            return the_time_delta_coefficient_ * (the_Najl_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) / (a_vpar_delta * a_vpar_delta) -
                                                  the_Najl_(the_batch_index, ivpar, 1, jpol, ipol, the_batch_element_index) / (a_vpar_delta * 2.0));
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::MiddleDiagonal(OrdinalType the_batch_index,
                                                      OrdinalType ivpar,
                                                      OrdinalType ipol,
                                                      OrdinalType jpol,
                                                      OrdinalType the_batch_element_index) const {
            // NOTE: vparg_(ivpar) - vparg_(ivpar + 1) would overflow
            const auto             a_vpar_delta = vparg_(1) - vparg_(0);
            const ComputeFloatType the_value = the_time_delta_coefficient_ * (the_Najl_(the_batch_index, ivpar, 0, jpol, ipol, the_batch_element_index) -
                                                                              (the_Najl_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) * 2.0) / (a_vpar_delta * a_vpar_delta));


            // the_matrix = I - dt/4 * T

            // TODO(Etienne M): On CPU this branch should get moved up
            // in the loop and on GPU I hope it does not cost too much.
            // Else we have to launch another kernel on the diagonal.
            // TODO(Etienne M): What about branch divergence.

            return jpol == ipol ?
                       the_value + 1.0 :
                       the_value;
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::UpperDiagonal(OrdinalType the_batch_index,
                                                     OrdinalType ivpar,
                                                     OrdinalType ipol,
                                                     OrdinalType jpol,
                                                     OrdinalType the_batch_element_index) const {
            // NOTE: vparg_(ivpar) - vparg_(ivpar + 1) would overflow
            const auto a_vpar_delta = vparg_(1) - vparg_(0);
            return the_time_delta_coefficient_ * (the_Najl_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) / (a_vpar_delta * a_vpar_delta) +
                                                  the_Najl_(the_batch_index, ivpar, 1, jpol, ipol, the_batch_element_index) / (a_vpar_delta * 2.0));
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION void
        MatrixSampler<View6D, View1D>::AllDiagonals(OrdinalType       the_batch_index,
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
