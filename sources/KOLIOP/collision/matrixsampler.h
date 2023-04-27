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
                          ComputeFloatType a_vpar_delta);

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

        protected:
            // TODO(Etienne M): having a copy may not be ideal. Maybe we should
            // pass everything the the sampling function.
            View6D           the_Nia_;
            ComputeFloatType the_coefficients_[2];
        };


        ////////////////////////////////////////////////////////////////////////
        // MatrixSampler methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename View6D>
        MatrixSampler<View6D>::MatrixSampler(const View6D&    the_Nia,
                                             ComputeFloatType a_vpar_delta)
            : the_Nia_{the_Nia}
            , the_coefficients_{1.0 / (a_vpar_delta * 2.0),
                                1.0 / (a_vpar_delta * a_vpar_delta)} {
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
            return the_Nia_(the_batch_index, ivpar, 0, jpol, ipol, the_batch_element_index) * the_coefficients_[0] -
                   the_Nia_(the_batch_index, ivpar, 2, jpol, ipol, the_batch_element_index) * the_coefficients_[1] * 2.0;
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

    } // namespace collision
} // namespace koliop

#endif
