#ifndef KOLIOP_COLLISION_POLICY_H
#define KOLIOP_COLLISION_POLICY_H

#include "KOLIOP/internal/configuration.h"


namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // CollisionPolicy definition
        ////////////////////////////////////////////////////////////////////////

        /// TODO(Etienne M): Parametrize this thing based on CMake options ?
        ///
        struct CollisionPolicy {
        public:
            using FloatType = Flo64;

            // TODO(Etienne M): Different execution spaces, appropriate view
            // types etc..
        };

    } // namespace collision
} // namespace koliop

#endif
