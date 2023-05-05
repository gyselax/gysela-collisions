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
            using ScalarBlockTridiagonalByBlockMatrixType = Constant::ScalarBlockTridiagonalByBlockMatrixType;
            using VectorBlockTridiagonalByBlockMatrixType = Constant::VectorBlockTridiagonalByBlockMatrixType;

        public:
            Najl(OrdinalType the_vpar_extent);

            /// @param the_Najl respects the format of
            /// ScalarBlockTridiagonalByBlockMatrixType.
            void Compute(ComputeFloatType* the_Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            koliop::utility::View::MDD2<ComputeFloatType> C_j_;  // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType> C_jl_; // TODO(Etienne M): constant extent.

            OrdinalType the_vpar_extent_;
        };

    } // namespace collision
} // namespace koliop

#endif
