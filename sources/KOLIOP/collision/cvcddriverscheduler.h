// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_CVCDDRIVERSCHEDULER_H
#define KOLIOP_COLLISION_CVCDDRIVERSCHEDULER_H

#include <vector>

#include "KOLIOP/collision/cvcddrivereraser.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        // NOTE: Ugly declaration..
        class Operator;

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverScheduler definition
        ////////////////////////////////////////////////////////////////////////

        class CvCdDriverScheduler {
        public:
            struct CvCdDriverConfiguration {
            public:
                CvCdDriverEraser                   the_cvcd_driver_;
                OrdinalType                        the_block_first_r_;
                OrdinalType                        the_block_first_theta_;
                OrdinalType                        the_block_first_phi_;
                OrdinalType                        the_block_r_extent_;
                OrdinalType                        the_block_theta_extent_;
                OrdinalType                        the_block_phi_extent_;
                Constant::DeviceExecutionSpaceType the_execution_space_;
            };

        public:
            void Plan(const Operator& an_operator,
                      std::size_t     the_preset_block_extent_count,
                      const OrdinalType (*the_preset_block_extent)[3]);

            void ScheduleExecution(const Operator&                                an_operator,
                                   ComputeFloatType                               dt,
                                   OrdinalType                                    the_species_index,
                                   koliop::utility::View::InternalInputL6<Flo64>& f5d_s);

        protected:
            std::vector<CvCdDriverConfiguration> the_cvcd_drivers_configurations_;
        };

    } // namespace collision
} // namespace koliop

#endif
