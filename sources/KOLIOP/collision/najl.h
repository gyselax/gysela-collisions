// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_NAJL_H
#define KOLIOP_COLLISION_NAJL_H

#include <Kokkos_Macros.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Najl definition
        ////////////////////////////////////////////////////////////////////////

        class Najl {
        public:
            Najl();

            template <typename View6D>
            void Compute(View6D&& the_Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            koliop::utility::View::MDD2<ComputeFloatType> C_j_;  // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType> C_jl_; // TODO(Etienne M): constant extent.
        };

    } // namespace collision
} // namespace koliop

#endif
