// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        void Operator::Apply(ComputeFloatType dt,
                             Flo64*           f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                                       species_extent_);

            // NOTE: Stream usage semantic change
            Fence(Constant::kDefaultExecutionSpacePoolIndex);
#if !defined(KOLIOP_CONSTANT_MEMORY_USAGE)
            cvcd_drivers_scheduler_.AllocateScratchMemory(*this);
#endif
            for(OrdinalType species_index = 0; species_index < species_extent_; ++species_index) {
                cvcd_drivers_scheduler_.ScheduleExecution(*this, dt, species_index, a_device_visible_f5d_s);
            }

            // NOTE: Stream usage semantic change. We must also ensure the
            // computation ended before we free.
            FenceAll();

#if !defined(KOLIOP_CONSTANT_MEMORY_USAGE)
            cvcd_drivers_scheduler_.ReleaseScratchMemory(*this);
#endif

            koliop::utility::Copy::ViewToRawData<koliop::utility::View::InternalInputL6<Flo64>,
                                                 koliop::utility::View::ExternalInputL6<Flo64>>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                a_device_visible_f5d_s,
                                                                                                f5d_s,
                                                                                                mu_extent_, vpar_extent_,
                                                                                                tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                species_extent_);
        }

    } // namespace collision
} // namespace koliop
