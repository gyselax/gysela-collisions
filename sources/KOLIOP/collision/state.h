// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>


namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // State definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Policy>
        class State {
        public:
            using PolicyType = Policy;

        public:
        protected:
            Kokkos::View<typename PolicyType::FloatType> a_thing_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
