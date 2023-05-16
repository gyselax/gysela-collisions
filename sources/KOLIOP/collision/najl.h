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
            Najl(OrdinalType                                          pSpecies_id,
                 OrdinalType                                          the_vpar_extent,
                 OrdinalType                                          the_species_extent,
                 const Flo64*                                         comb_mat,
                 koliop::utility::View::MDD1<const ComputeFloatType>& hat_As,
                 koliop::utility::View::MDD1<const ComputeFloatType>& hat_Zs,
                 ExecutionSpaceType                                   an_execution_space);

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

            // Constant, externally computed values

            OrdinalType pSpecies_id_;
            OrdinalType the_vpar_extent_;
            OrdinalType the_species_extent_;

            koliop::utility::View::MDD2<const ComputeFloatType> comb_mat_; // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD1<const ComputeFloatType> hat_As_;
            koliop::utility::View::MDD1<const ComputeFloatType> hat_Zs_;

            // Constant, internally computed values

            koliop::utility::View::MDD2<ComputeFloatType> C_j_;  // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType> C_jl_; // TODO(Etienne M): constant extent.

            // Miscellaneous tooling

            ExecutionSpaceType the_execution_space_;
        };

    } // namespace collision
} // namespace koliop

#endif
