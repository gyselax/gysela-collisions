// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/cvcddriverscheduler.h"

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {
        namespace detail {

            static inline void
            PrepareBlockPosition(const OrdinalType                                          local_domain_r_extent,
                                 const OrdinalType                                          local_domain_theta_extent,
                                 const OrdinalType                                          local_domain_phi_extent,
                                 std::vector<CvCdDriverScheduler::CvCdDriverConfiguration>& cvcd_drivers_configurations,
                                 std::size_t                                                preset_block_extent_count,
                                 const OrdinalType (*preset_block_extent)[3]) {

                auto       first_block_extent = preset_block_extent;
                const auto last_block_extent  = first_block_extent + preset_block_extent_count;

                for(; first_block_extent != last_block_extent; ++first_block_extent) {
                    const auto& current_block_extent = *first_block_extent;

                    const auto current_r_block_extent     = current_block_extent[0];
                    const auto current_theta_block_extent = current_block_extent[1];
                    const auto current_phi_block_extent   = current_block_extent[2];

                    KOLIOP_ASSERT(current_r_block_extent > 0 &&
                                  current_theta_block_extent > 0 &&
                                  current_phi_block_extent > 0);

                    const auto block_count_in_r     = local_domain_r_extent / current_r_block_extent;
                    const auto block_count_in_theta = local_domain_theta_extent / current_theta_block_extent;
                    const auto block_count_in_phi   = local_domain_phi_extent / current_phi_block_extent;

                    if(block_count_in_r > 0 &&
                       block_count_in_theta > 0 &&
                       block_count_in_phi > 0) {

                        OrdinalType block_first_phi = local_domain_phi_extent;
                        for(OrdinalType block_index_in_phi = 0; block_index_in_phi < block_count_in_phi; ++block_index_in_phi) {
                            block_first_phi -= current_phi_block_extent;

                            OrdinalType block_first_theta = local_domain_theta_extent;
                            for(OrdinalType block_index_in_theta = 0; block_index_in_theta < block_count_in_theta; ++block_index_in_theta) {
                                block_first_theta -= current_theta_block_extent;

                                OrdinalType block_first_r = local_domain_r_extent;
                                for(OrdinalType block_index_in_r = 0; block_index_in_r < block_count_in_r; ++block_index_in_r) {
                                    block_first_r -= current_r_block_extent;

                                    auto& a_driver_configuration = cvcd_drivers_configurations.emplace_back();

                                    a_driver_configuration.block_first_r_     = block_first_r;
                                    a_driver_configuration.block_first_theta_ = block_first_theta;
                                    a_driver_configuration.block_first_phi_   = block_first_phi;

                                    a_driver_configuration.block_r_extent_     = current_r_block_extent;
                                    a_driver_configuration.block_theta_extent_ = current_theta_block_extent;
                                    a_driver_configuration.block_phi_extent_   = current_phi_block_extent;
                                }
                            }
                        }

                        // NOTE: Once a block that fits is found, because the domain's dimension are power of two, we
                        // know that we wont need an other block size. We can exit early.

                        KOLIOP_ASSERT((local_domain_r_extent - block_count_in_r * current_r_block_extent) == 0 &&
                                      (local_domain_theta_extent - block_count_in_theta * current_theta_block_extent) == 0 &&
                                      (local_domain_phi_extent - block_count_in_phi * current_phi_block_extent) == 0);

                        break;
                    }

                    // NOTE: The block wont fit, go to the next (smaller) block
                    // size.
                }

                KOLIOP_ASSERT(!cvcd_drivers_configurations.empty());
            }

            // NOTE: Sort of factory. I put in into an other translation unit
            // to mitigate the long compilation time (yay for c++ templates).
            void CreateCvCdDriver(const Operator&   an_operator,
                                  CvCdDriverEraser& a_cvcd_driver,
                                  OrdinalType       block_r_extent,
                                  OrdinalType       block_theta_extent,
                                  OrdinalType       block_phi_extent);

        } // namespace detail

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverScheduler methods definition
        ////////////////////////////////////////////////////////////////////////

        void CvCdDriverScheduler::Plan(const Operator& an_operator,
                                       std::size_t     preset_block_extent_count,
                                       const OrdinalType (*preset_block_extent)[3]) {

            // NOTE: We assume preset_block_extent are sorted from larger to
            // smaller. This will not affect the correctness of the algorithm as
            // long as the minimum block size (of which all block are a
            // multiple) is present in the set. that said, this could affect the
            // performance of the cvcd driver. For instance if we use only small
            // block size where we could have used bigger ones.

            cvcd_drivers_configurations_.clear();

            detail::PrepareBlockPosition(an_operator.r_extent_,
                                         an_operator.theta_extent_,
                                         an_operator.phi_extent_,
                                         cvcd_drivers_configurations_,
                                         preset_block_extent_count,
                                         preset_block_extent);

            // NOTE:
            // - A larger block take more memory.
            // - By definition, we only need one block size to pave the plane.
            // - We have at most kDisjointExecutionSpaceCount disjoint execution
            // spaces.
            // - Drivers using different execution space MUST have disjoint
            // state!
            // - Knowing that, we can allocate only kDisjointExecutionSpaceCount
            // drivers (of the same size) and simply ventilate theses driver on
            // the execution spaces. We use shared pointer to share the drivers.
            // We can guarantee that at most something proportionate to
            // kDisjointExecutionSpaceCount * BlockSize will be allocated.
            // Shared instances of a driver share the same execution space. This
            // basically serialize the work from the POV of a driver preventing
            // race condition.

            const auto& execution_space_pool = an_operator.execution_space_pool_;

            // The first (at most) execution_space_pool.size() get a freshly
            // allocated CvCdDriver and an execution space. The other block will
            // be copies of these ones.
            const OrdinalType original_block_count = std::min(cvcd_drivers_configurations_.size(),
                                                              execution_space_pool.size());

            for(OrdinalType i = 0; i < original_block_count; ++i) {
                auto& a_driver_configuration = cvcd_drivers_configurations_[i];

                detail::CreateCvCdDriver(an_operator,
                                         a_driver_configuration.cvcd_driver_,
                                         a_driver_configuration.block_r_extent_,
                                         a_driver_configuration.block_theta_extent_,
                                         a_driver_configuration.block_phi_extent_);

                // The modulo would be redundant thanks to
                // original_block_count.
                const OrdinalType execution_space_index = i /* % execution_space_pool.size() */;

                a_driver_configuration.execution_space_ = execution_space_pool[execution_space_index];
            }

            for(OrdinalType i = original_block_count;
                i < static_cast<OrdinalType>(cvcd_drivers_configurations_.size());
                ++i) {
                auto&       a_driver_configuration           = cvcd_drivers_configurations_[i];
                const auto& an_original_driver_configuration = cvcd_drivers_configurations_[i % original_block_count];

                // NOTE: We copy, but the CvCdDriverEraser uses shared pointer
                // which *prevents* deep copying.

                // NOTE: We copy the execution space and the driver but NOT
                // block_first_r_, block_first_theta_ and
                // block_first_phi_. These are and must stay different.
                a_driver_configuration.cvcd_driver_     = an_original_driver_configuration.cvcd_driver_;
                a_driver_configuration.execution_space_ = an_original_driver_configuration.execution_space_;
            }
        }

        void CvCdDriverScheduler::ScheduleExecution(const Operator&                                an_operator,
                                                    ComputeFloatType                               dt,
                                                    OrdinalType                                    species_index,
                                                    koliop::utility::View::InternalInputL6<Flo64>& f5d_s) {
            for(auto&& a_driver_configuration : cvcd_drivers_configurations_) {
                a_driver_configuration.cvcd_driver_->Apply(an_operator,
                                                           a_driver_configuration.execution_space_,
                                                           a_driver_configuration.block_first_r_,
                                                           a_driver_configuration.block_first_theta_,
                                                           a_driver_configuration.block_first_phi_,
                                                           dt,
                                                           species_index,
                                                           f5d_s);
            }
        }

    } // namespace collision
} // namespace koliop
