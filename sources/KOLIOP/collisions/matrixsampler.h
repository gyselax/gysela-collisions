// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_MATRIXSAMPLER_H
#define KOLIOP_COLLISION_MATRIXSAMPLER_H

#include <Kokkos_Macros.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace collisions {

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
            MatrixSampler(const View6D& Najl,
                          const View1D& vpar_delta);

            KOKKOS_INLINE_FUNCTION ComputeFloatType&
            DeltaTime();

            KOKKOS_INLINE_FUNCTION const ComputeFloatType&
            DeltaTime() const;

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            operator()(OrdinalType batch_index,
                       OrdinalType ivpar,
                       OrdinalType diagonal,
                       OrdinalType ipol,
                       OrdinalType jpol,
                       OrdinalType batch_element_index) const;

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            LowerDiagonal(OrdinalType batch_index,
                          OrdinalType ivpar,
                          OrdinalType ipol,
                          OrdinalType jpol,
                          OrdinalType batch_element_index) const;

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            MiddleDiagonal(OrdinalType batch_index,
                           OrdinalType ivpar,
                           OrdinalType ipol,
                           OrdinalType jpol,
                           OrdinalType batch_element_index) const;

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            UpperDiagonal(OrdinalType batch_index,
                          OrdinalType ivpar,
                          OrdinalType ipol,
                          OrdinalType jpol,
                          OrdinalType batch_element_index) const;

            KOKKOS_INLINE_FUNCTION void
            AllDiagonals(OrdinalType       batch_index,
                         OrdinalType       ivpar,
                         OrdinalType       ipol,
                         OrdinalType       jpol,
                         OrdinalType       batch_element_index,
                         ComputeFloatType& lower_diagonal,
                         ComputeFloatType& middle_diagonal,
                         ComputeFloatType& upper_diagonal) const;

        protected:
            // TODO(Etienne M): having a copy may not be ideal. Maybe we should
            // pass everything to the sampling function.
            View6D           Najl_;
            View1D           vpar_delta_;
            ComputeFloatType time_delta_coefficient_;
        };


        ////////////////////////////////////////////////////////////////////////
        // MatrixSampler methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename View6D,
                  typename View1D>
        MatrixSampler<View6D, View1D>::MatrixSampler(const View6D& Najl,
                                                     const View1D& vpar_delta)
            : Najl_{Najl}
            , vpar_delta_{vpar_delta}
            , time_delta_coefficient_{} {
            // EMPTY
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType&
        MatrixSampler<View6D, View1D>::DeltaTime() {
            return time_delta_coefficient_;
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION const ComputeFloatType&
        MatrixSampler<View6D, View1D>::DeltaTime() const {
            return time_delta_coefficient_;
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::operator()(OrdinalType batch_index,
                                                  OrdinalType ivpar,
                                                  OrdinalType diagonal,
                                                  OrdinalType ipol,
                                                  OrdinalType jpol,
                                                  OrdinalType batch_element_index) const {
            if(diagonal == 0) {
                return LowerDiagonal(batch_index,
                                     ivpar,
                                     ipol,
                                     jpol,
                                     batch_element_index);
            } else if(diagonal == 1) {
                return MiddleDiagonal(batch_index,
                                      ivpar,
                                      ipol,
                                      jpol,
                                      batch_element_index);
            } else if(diagonal == 2) {
                return UpperDiagonal(batch_index,
                                     ivpar,
                                     ipol,
                                     jpol,
                                     batch_element_index);
            } else {
                // GPU assert ?
                return -1.0;
            }
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::LowerDiagonal(OrdinalType batch_index,
                                                     OrdinalType ivpar,
                                                     OrdinalType ipol,
                                                     OrdinalType jpol,
                                                     OrdinalType batch_element_index) const {
            const auto a_vpar_delta = vpar_delta_(ivpar);

            return DeltaTime() * (Najl_(batch_index, ivpar, 2, ipol, jpol, batch_element_index) / (a_vpar_delta * a_vpar_delta) -
                                  Najl_(batch_index, ivpar, 1, ipol, jpol, batch_element_index) / (a_vpar_delta * 2.0));
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::MiddleDiagonal(OrdinalType batch_index,
                                                      OrdinalType ivpar,
                                                      OrdinalType ipol,
                                                      OrdinalType jpol,
                                                      OrdinalType batch_element_index) const {
            const auto a_vpar_delta = vpar_delta_(ivpar);

            const ComputeFloatType value = DeltaTime() * (Najl_(batch_index, ivpar, 0, ipol, jpol, batch_element_index) -
                                                          (Najl_(batch_index, ivpar, 2, ipol, jpol, batch_element_index) * 2.0) / (a_vpar_delta * a_vpar_delta));


            // matrix = I - dt/4 * T

            // NOTE: On CPU this branch should get moved up in the loop and on
            // GPU I hope it does not cost too much. Else we have to launch
            // another kernel on the diagonal.

            return jpol == ipol ?
                       value + 1.0 :
                       value;
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION ComputeFloatType
        MatrixSampler<View6D, View1D>::UpperDiagonal(OrdinalType batch_index,
                                                     OrdinalType ivpar,
                                                     OrdinalType ipol,
                                                     OrdinalType jpol,
                                                     OrdinalType batch_element_index) const {
            const auto a_vpar_delta = vpar_delta_(ivpar);

            return DeltaTime() * (Najl_(batch_index, ivpar, 2, ipol, jpol, batch_element_index) / (a_vpar_delta * a_vpar_delta) +
                                  Najl_(batch_index, ivpar, 1, ipol, jpol, batch_element_index) / (a_vpar_delta * 2.0));
        }

        template <typename View6D,
                  typename View1D>
        KOKKOS_INLINE_FUNCTION void
        MatrixSampler<View6D, View1D>::AllDiagonals(OrdinalType       batch_index,
                                                    OrdinalType       ivpar,
                                                    OrdinalType       ipol,
                                                    OrdinalType       jpol,
                                                    OrdinalType       batch_element_index,
                                                    ComputeFloatType& lower_diagonal,
                                                    ComputeFloatType& middle_diagonal,
                                                    ComputeFloatType& upper_diagonal) const {
            lower_diagonal = LowerDiagonal(batch_index,
                                           ivpar,
                                           ipol,
                                           jpol,
                                           batch_element_index);

            middle_diagonal = MiddleDiagonal(batch_index,
                                             ivpar,
                                             ipol,
                                             jpol,
                                             batch_element_index);

            upper_diagonal = UpperDiagonal(batch_index,
                                           ivpar,
                                           ipol,
                                           jpol,
                                           batch_element_index);
        }

    } // namespace collisions
} // namespace koliop

#endif
