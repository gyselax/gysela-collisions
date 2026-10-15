// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_CVCDDRIVERSCHEDULER_H
#define KOLIOP_COLLISION_CVCDDRIVERSCHEDULER_H

#include <vector>

#include "KOLIOP/collisions/cvcddrivereraser.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collisions {

        class Operator;

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverScheduler definition
        ////////////////////////////////////////////////////////////////////////

        class CvCdDriverScheduler {
        public:
            struct CvCdDriverConfiguration {
            public:
                CvCdDriverEraser                   cvcd_driver_;
                OrdinalType                        block_first_tor1_;
                OrdinalType                        block_first_tor2_;
                OrdinalType                        block_first_tor3_;
                OrdinalType                        block_tor1_extent_;
                OrdinalType                        block_tor2_extent_;
                OrdinalType                        block_tor3_extent_;
                Constant::DeviceExecutionSpaceType execution_space_;
            };

        public:
            void Plan(const Operator& an_operator,
                      std::size_t     preset_block_extent_count,
                      const OrdinalType (*preset_block_extent)[3]);

            void AllocateScratchMemory(const Operator& an_operator);
            void ReleaseScratchMemory(const Operator& an_operator);

            void ScheduleExecution(const Operator&                           an_operator,
                                   ComputeFloatType                          dt,
                                   OrdinalType                               species_index,
                                   const koliop::utility::View::UDL6<Flo64>& f5d_s);

        protected:
            std::vector<CvCdDriverConfiguration> cvcd_drivers_configurations_;
        };

    } // namespace collisions
} // namespace koliop

#endif
