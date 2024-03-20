// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_SCOPED_H
#define KOLIOP_COLLISION_SCOPED_H

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // ScopedFenceHandler definition
        ////////////////////////////////////////////////////////////////////////

        template <bool should_fence>
        class ScopedFenceHandler {
        public:
        public:
            ScopedFenceHandler(const Operator* an_operator)
                : the_operator_{an_operator} {
                if constexpr(should_fence) {
                    the_operator_->Fence(Constant::kDefaultExecutionSpacePoolIndex);
                }
            }

            ~ScopedFenceHandler() {
                if constexpr(should_fence) {
                    // The many streams used for the operation now need to be synced
                    // before we start copying.
                    the_operator_->FenceAll();
                }
            }

        protected:
            const Operator* the_operator_;
        };

    } // namespace collision
} // namespace koliop

#endif
