// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "KOLIOP/collisions.h"
#include "KOLIOP/collisions/constants.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collisions {

        void Operator::Apply(ComputeFloatType                          dt,
                             const koliop::utility::View::MDL6<Flo64>& f5d_s) {
            // NOTE: Stream usage semantic change
            Fence(Constant::kDefaultExecutionSpacePoolIndex);
#if !defined(KOLIOP_CONSTANT_MEMORY_USAGE)
            cvcd_drivers_scheduler_.AllocateScratchMemory(*this);
#endif
            for(OrdinalType species_index = 0; species_index < species_extent_; ++species_index) {
                cvcd_drivers_scheduler_.ScheduleExecution(*this, dt, species_index, f5d_s);
            }

            // NOTE: Stream usage semantic change. We must also ensure the
            // computation ended before we free.
            FenceAll();

#if !defined(KOLIOP_CONSTANT_MEMORY_USAGE)
            cvcd_drivers_scheduler_.ReleaseScratchMemory(*this);
#endif
        }

    } // namespace collisions
} // namespace koliop
