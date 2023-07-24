// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {


        template <typename View6D>
        static inline void
        DoApply(Operator&        an_operator,
                ComputeFloatType dt,
                View6D&&         f5d_s) {
            // TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
            // we have to do something similar.

            const auto& the_r_extent     = an_operator.the_r_extent_;
            const auto& the_theta_extent = an_operator.the_theta_extent_;
            const auto& the_phi_extent   = an_operator.the_phi_extent_;

            auto& the_cvcd_drivers = an_operator.the_cvcd_drivers_;

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            // TODO(Etienne M): Make the Cvcd opaque to the operator, reducing
            // the dependencies and recompilation duration.

            // TODO(Etienne M): Blocking idea, use a recursive approach where we
            // try to fit the biggest block size to process the domain, and if
            // not possible we fallback to a smaller size. Then, we process the
            // potential remaining grid point recursively. Supposing we do not
            // template CvCd on the block size anymore, we could reuse them for
            // smaller blocs.

            // TODO(Etienne M): investigate the issue with full fence during
            // view deallocation.

            using CvCdDriver = decltype(an_operator.the_cvcd_drivers_)::value_type;

            KOLIOP_ASSERT(the_cvcd_drivers.size() == Constant::kDisjointExecutionSpaceCount);

            uint64_t the_cvcd_operation_index = 0;

            for(OrdinalType the_block_first_phi = 0;
                the_block_first_phi < the_phi_extent;
                the_block_first_phi += CvCdDriver::kBlockExtentPhi) {
                for(OrdinalType the_block_first_theta = 0;
                    the_block_first_theta < the_theta_extent;
                    the_block_first_theta += CvCdDriver::kBlockExtentTheta) {
                    for(OrdinalType the_block_first_r = 0;
                        the_block_first_r < the_r_extent;
                        the_block_first_r += CvCdDriver::kBlockExtentR) {
                        const uint64_t the_driver_index = the_cvcd_operation_index % Constant::kDisjointExecutionSpaceCount;

                        // NOTE: We make sure that the execution space index and
                        // the driver index match so that no execution space
                        // (aka. stream) can start using a driver before a
                        // potential, previous execution space ended using it.
                        const uint64_t the_execution_space_index = the_driver_index;

                        the_cvcd_drivers[the_driver_index].Apply(an_operator,
                                                                 the_execution_space_pool[the_execution_space_index],
                                                                 the_block_first_r,
                                                                 the_block_first_theta,
                                                                 the_block_first_phi,
                                                                 dt,
                                                                 f5d_s);

                        ++the_cvcd_operation_index;
                    }
                }
            }

            // TODO(Etienne M): Do we full fence or wait for all the streams ?
            // Fence is a lot cheaper it seems. Though if Gysela executes other
            // GPU kernel while koliop works, this could slow things done.

            Kokkos::fence();

            // for(const auto& an_execution_space : the_execution_space_pool) {
            //     an_execution_space.fence();
            // }
        }

        void Operator::Apply(ComputeFloatType dt,
                             Flo64*           f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            DoApply(*this, dt, a_device_visible_f5d_s);

            koliop::utility::Copy::ViewToRawData<koliop::utility::View::InternalInputL6<Flo64>,
                                                 koliop::utility::View::ExternalInputL6<Flo64>>(a_device_visible_f5d_s,
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
