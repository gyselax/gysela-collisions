// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/cvcddriverscheduler.h"

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {
        namespace detail {

            static inline void
            PrepareBlockPosition(const OrdinalType                                          the_local_domain_r_extent,
                                 const OrdinalType                                          the_local_domain_theta_extent,
                                 const OrdinalType                                          the_local_domain_phi_extent,
                                 std::vector<CvCdDriverScheduler::CvCdDriverConfiguration>& the_cvcd_drivers_configurations,
                                 std::size_t                                                the_preset_block_extent_count,
                                 const OrdinalType (*the_preset_block_extent)[3]) {

                auto       the_first_block_extent = the_preset_block_extent;
                const auto the_last_block_extent  = the_first_block_extent + the_preset_block_extent_count;

                for(; the_first_block_extent != the_last_block_extent; ++the_first_block_extent) {
                    const auto& the_current_block_extent = *the_first_block_extent;

                    const auto the_current_r_block_extent     = the_current_block_extent[0];
                    const auto the_current_theta_block_extent = the_current_block_extent[1];
                    const auto the_current_phi_block_extent   = the_current_block_extent[2];

                    KOLIOP_ASSERT(the_current_r_block_extent > 0 &&
                                  the_current_theta_block_extent > 0 &&
                                  the_current_phi_block_extent > 0);

                    const auto the_block_count_in_r     = the_local_domain_r_extent / the_current_r_block_extent;
                    const auto the_block_count_in_theta = the_local_domain_theta_extent / the_current_theta_block_extent;
                    const auto the_block_count_in_phi   = the_local_domain_phi_extent / the_current_phi_block_extent;

                    if(the_block_count_in_r > 0 &&
                       the_block_count_in_theta > 0 &&
                       the_block_count_in_phi > 0) {

                        OrdinalType the_block_first_phi = the_local_domain_phi_extent;
                        for(OrdinalType the_block_index_in_phi = 0; the_block_index_in_phi < the_block_count_in_phi; ++the_block_index_in_phi) {
                            the_block_first_phi -= the_current_phi_block_extent;

                            OrdinalType the_block_first_theta = the_local_domain_theta_extent;
                            for(OrdinalType the_block_index_in_theta = 0; the_block_index_in_theta < the_block_count_in_theta; ++the_block_index_in_theta) {
                                the_block_first_theta -= the_current_theta_block_extent;

                                OrdinalType the_block_first_r = the_local_domain_r_extent;
                                for(OrdinalType the_block_index_in_r = 0; the_block_index_in_r < the_block_count_in_r; ++the_block_index_in_r) {
                                    the_block_first_r -= the_current_r_block_extent;

                                    auto& a_driver_configuration = the_cvcd_drivers_configurations.emplace_back();

                                    a_driver_configuration.the_block_first_r_     = the_block_first_r;
                                    a_driver_configuration.the_block_first_theta_ = the_block_first_theta;
                                    a_driver_configuration.the_block_first_phi_   = the_block_first_phi;

                                    a_driver_configuration.the_block_r_extent_     = the_current_r_block_extent;
                                    a_driver_configuration.the_block_theta_extent_ = the_current_theta_block_extent;
                                    a_driver_configuration.the_block_phi_extent_   = the_current_phi_block_extent;
                                }
                            }
                        }

                        // NOTE: Once a block that fits is found, because the domain's dimension are power of two, we
                        // know that we wont need an other block size. We can exit early.

                        KOLIOP_ASSERT((the_local_domain_r_extent - the_block_count_in_r * the_current_r_block_extent) == 0 &&
                                      (the_local_domain_theta_extent - the_block_count_in_theta * the_current_theta_block_extent) == 0 &&
                                      (the_local_domain_phi_extent - the_block_count_in_phi * the_current_phi_block_extent) == 0);

                        break;
                    }

                    // NOTE: The block wont fit, go to the next (smaller) block
                    // size.
                }
            }

            // NOTE: Sort of factory. I put in into an other translation unit
            // to mitigate the long compilation time (yay for c++ templates).
            void CreateCvCdDriver(const Operator&   an_operator,
                                  CvCdDriverEraser& a_cvcd_driver,
                                  OrdinalType       the_block_r_extent,
                                  OrdinalType       the_block_theta_extent,
                                  OrdinalType       the_block_phi_extent);

        } // namespace detail

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverScheduler methods definition
        ////////////////////////////////////////////////////////////////////////

        void CvCdDriverScheduler::Plan(const Operator& an_operator,
                                       std::size_t     the_preset_block_extent_count,
                                       const OrdinalType (*the_preset_block_extent)[3]) {

            // NOTE: We assume the_preset_block_extent are sorted from larger to
            // smaller. This will not affect the correctness of the algorithm as
            // long as the minimum block size (of which all block are a
            // multiple) is present in the set. that said, this could affect the
            // performance of the cvcd driver. For instance if we use only small
            // block size where we could have used bigger ones.

            the_cvcd_drivers_configurations_.clear();

            detail::PrepareBlockPosition(an_operator.the_r_extent_,
                                         an_operator.the_theta_extent_,
                                         an_operator.the_phi_extent_,
                                         the_cvcd_drivers_configurations_,
                                         the_preset_block_extent_count,
                                         the_preset_block_extent);

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
            // We must make sure that only one execution space used for a given
            // execution space. So shared instances of a driver share the same
            // execution space. This basically serialize the work from the POV
            // of a driver preventing race condition.

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            // The first (at most) the_execution_space_pool.size() get a freshly
            // allocated CvCdDriver and an execution space. The other block will
            // be copies of these ones.
            const OrdinalType the_original_block_count = std::min(the_cvcd_drivers_configurations_.size(),
                                                                  the_execution_space_pool.size());

            for(OrdinalType i = 0; i < the_original_block_count; ++i) {
                auto& a_driver_configuration = the_cvcd_drivers_configurations_[i];

                detail::CreateCvCdDriver(an_operator,
                                         a_driver_configuration.the_cvcd_driver_,
                                         a_driver_configuration.the_block_r_extent_,
                                         a_driver_configuration.the_block_theta_extent_,
                                         a_driver_configuration.the_block_phi_extent_);

                // The modulo would be redundant thanks to
                // the_original_block_count.
                const OrdinalType the_execution_space_index = i /* % the_execution_space_pool.size() */;

                a_driver_configuration.the_execution_space_ = the_execution_space_pool[the_execution_space_index];
            }

            for(OrdinalType i = the_original_block_count;
                i < static_cast<OrdinalType>(the_cvcd_drivers_configurations_.size());
                ++i) {
                auto&       a_driver_configuration           = the_cvcd_drivers_configurations_[i];
                const auto& an_original_driver_configuration = the_cvcd_drivers_configurations_[i % the_original_block_count];

                // NOTE: We copy, but the CvCdDriverEraser uses shared pointer
                // which *prevents* deep copying.

                // NOTE: We copy the execution space and the driver but NOT
                // the_block_first_r_, the_block_first_theta_ and
                // the_block_first_phi_. These are and must stay different.
                a_driver_configuration.the_cvcd_driver_     = an_original_driver_configuration.the_cvcd_driver_;
                a_driver_configuration.the_execution_space_ = an_original_driver_configuration.the_execution_space_;
            }
        }

        void CvCdDriverScheduler::ScheduleExecution(const Operator&                                an_operator,
                                                    ComputeFloatType                               dt,
                                                    koliop::utility::View::InternalInputL6<Flo64>& f5d_s) {
            for(auto&& a_driver_configuration : the_cvcd_drivers_configurations_) {
                a_driver_configuration.the_cvcd_driver_->Apply(an_operator,
                                                               a_driver_configuration.the_execution_space_,
                                                               a_driver_configuration.the_block_first_r_,
                                                               a_driver_configuration.the_block_first_theta_,
                                                               a_driver_configuration.the_block_first_phi_,
                                                               dt,
                                                               f5d_s);
            }
        }

    } // namespace collision
} // namespace koliop
