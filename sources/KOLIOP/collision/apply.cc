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
                                                                               koliop::utility::View::InternalInputL6>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            // NOTE: WE must fence as Apply may not work in the same queue than
            // the other routines. We have to fence using the *general* fence
            // routine which fences only on the default execution space used by
            // all other kernel in the operator.
            Fence();

            {
                the_cvcd_drivers_scheduler_.ScheduleExecution(*this, dt, a_device_visible_f5d_s);
            }

            // TODO(Etienne M): Do we full fence or wait for all the streams ?
            // Fence is a lot cheaper it seems. Though if Gysela executes other
            // GPU kernel while koliop works, this could slow things done.

            // NOTE: This fence does not mean the user doesn't have to fence
            // after the operator returns. Indeed, the copies below may very
            // well be async.
            Kokkos::fence();

            // for(const auto& an_execution_space : the_execution_space_pool) {
            //     an_execution_space.fence();
            // }

            koliop::utility::Copy::ViewToRawData<koliop::utility::View::InternalInputL6<Flo64>,
                                                 koliop::utility::View::ExternalInputL6<Flo64>>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                a_device_visible_f5d_s,
                                                                                                f5d_s,
                                                                                                the_mu_extent_, the_vpar_extent_,
                                                                                                the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                the_species_extent_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
