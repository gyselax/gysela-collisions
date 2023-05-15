// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_NAJL_H
#define KOLIOP_COLLISION_NAJL_H

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Najl definition
        ////////////////////////////////////////////////////////////////////////

        class Najl {
        public:
            using ExecutionSpaceType = typename Constant::DeviceExecutionSpaceType;

            using ScalarBlockTridiagonalByBlockMatrixType = Constant::ScalarBlockTridiagonalByBlockMatrixType;
            using VectorBlockTridiagonalByBlockMatrixType = Constant::VectorBlockTridiagonalByBlockMatrixType;

        public:
            Najl(OrdinalType        the_vpar_extent,
                 const Flo64*       comb_mat,
                 ExecutionSpaceType an_execution_space);

            /// @param the_Najl respects the format of
            /// ScalarBlockTridiagonalByBlockMatrixType.
            void Compute(OrdinalType       the_block_first_r,
                         OrdinalType       the_block_first_theta,
                         OrdinalType       the_block_first_phi,
                         ComputeFloatType* the_Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            koliop::utility::View::MDD2<const ComputeFloatType> comb_mat_; // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD2<ComputeFloatType>       C_j_;      // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType>       C_jl_;     // TODO(Etienne M): constant extent.

            OrdinalType the_vpar_extent_;

            ExecutionSpaceType the_execution_space_;
        };

    } // namespace collision
} // namespace koliop

#endif
