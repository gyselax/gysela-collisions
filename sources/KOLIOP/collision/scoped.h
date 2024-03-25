// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_SCOPED_H
#define KOLIOP_COLLISION_SCOPED_H

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // ScopedFenceHandler definition
        ////////////////////////////////////////////////////////////////////////

        /// This scope guard ensure that if should_fence is true, we fence on
        /// kDefaultExecutionSpacePoolIndex during construction and on all
        /// streams on destruction.
        /// The use case is to ensure the data is correctly placed on the device
        /// or host before enqueuing stream memcpy. As an example, the device is
        /// the GPU and host the CPU, we need to wait before all the stream on
        /// which we enqueued the processing are emptied before the data being
        /// processed can be copied back to the device. This is a noop if
        /// should_fence is false.
        ///
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
