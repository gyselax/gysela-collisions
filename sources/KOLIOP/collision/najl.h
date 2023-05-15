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
            Najl(OrdinalType        pSpecies_id,
                 OrdinalType        the_vpar_extent,
                 const Flo64*       comb_mat,
                 ExecutionSpaceType an_execution_space);

            void Compute(OrdinalType                                          the_block_first_r,
                         OrdinalType                                          the_block_first_theta,
                         OrdinalType                                          the_block_first_phi,
                         const koliop::utility::View::MDD4<ComputeFloatType>& n_s,
                         const koliop::utility::View::MDD4<ComputeFloatType>& T_s,
                         const koliop::utility::View::MDD5<ComputeFloatType>& nu_ab,
                         Constant::ScalarBlockTridiagonalByBlockMatrixType&   the_Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            OrdinalType pSpecies_id_;
            OrdinalType the_vpar_extent_;

            koliop::utility::View::MDD2<const ComputeFloatType> comb_mat_; // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD2<ComputeFloatType>       C_j_;      // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType>       C_jl_;     // TODO(Etienne M): constant extent.

            ExecutionSpaceType the_execution_space_;
        };

    } // namespace collision
} // namespace koliop

#endif
