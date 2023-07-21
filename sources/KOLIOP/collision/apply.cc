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

            // TODO(Etienne M): Check what can be const in cvcd adn make it
            // const notably the view extent.

            // TODO(Etienne M): Make the Cvcd opaque to the operator, reducing
            // the dependencies and recompilation duration.

            // TODO(Etienne M): Blocking idea: On GPU and CPU we assume a large
            // block size and fill in the gaps using smaller block size.
            // We simply need to instantiate a smaller cvcd driver and
            // distribute the leftover work on these once the big block cvcd
            // driver handled most of the work.

            // TODO(Etienne M): investigate the issue with full fence during
            // view deallocation.

            using CvCdDriver = decltype(an_operator.the_cvcd_drivers_)::value_type;

            for(OrdinalType the_block_first_phi = 0;
                the_block_first_phi < the_phi_extent;
                the_block_first_phi += CvCdDriver::kBlockExtentPhi) {
                for(OrdinalType the_block_first_theta = 0;
                    the_block_first_theta < the_theta_extent;
                    the_block_first_theta += CvCdDriver::kBlockExtentTheta) {
                    for(OrdinalType the_block_first_r = 0;
                        the_block_first_r < the_r_extent;
                        the_block_first_r += CvCdDriver::kBlockExtentR) {
                        the_cvcd_drivers[0].Apply(an_operator,
                                                  the_execution_space_pool[0],
                                                  the_block_first_r,
                                                  the_block_first_theta,
                                                  the_block_first_phi,
                                                  dt,
                                                  f5d_s);
                    }
                }
            }

            // TODO(Etienne M): When views deallocated, there is a fence. Move
            // that allocations to the operator object ! Note that the matrix
            // sample also deallocate View.
            // NOTE: does the fence occurs when the storage is freed, or when to
            // View object's destructor is called ?
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
