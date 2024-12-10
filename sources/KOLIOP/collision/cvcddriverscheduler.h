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
                CvCdDriverEraser                   cvcd_driver_;
                OrdinalType                        block_first_r_;
                OrdinalType                        block_first_theta_;
                OrdinalType                        block_first_phi_;
                OrdinalType                        block_r_extent_;
                OrdinalType                        block_theta_extent_;
                OrdinalType                        block_phi_extent_;
                Constant::DeviceExecutionSpaceType execution_space_;
            };

        public:
            void Plan(const Operator& an_operator,
                      std::size_t     preset_block_extent_count,
                      const OrdinalType (*preset_block_extent)[3]);

            void ScheduleExecution(const Operator&                                an_operator,
                                   ComputeFloatType                               dt,
                                   OrdinalType                                    species_index,
                                   koliop::utility::View::InternalInputL6<Flo64>& f5d_s);

        protected:
            std::vector<CvCdDriverConfiguration> cvcd_drivers_configurations_;
        };

    } // namespace collision
} // namespace koliop

#endif
