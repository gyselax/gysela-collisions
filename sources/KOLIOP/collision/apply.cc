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

            // TODO(Etienne M): On GPU, should we just scale the block size ?
            // TODO(Etienne M): We use mask to avoid some computation, but
            // wouldn't it be better to compute for everything and only in the
            // end, reconstruct f based on the mask ?
            // TODO(Etienne M): On GPU we want big blocks and a single thread,
            // unparallelized host loop is enough to saturate the GPU. Would
            // this also work correctly on CPU ?


            CvCd<Constant::kBlockExtentR,
                 Constant::kBlockExtentTheta,
                 Constant::kBlockExtentPhi,
                 Constant::BatchTraitsType>
                the_cvcd_operator{an_operator.pSpecies_id_,
                                  an_operator.the_vpar_extent_,
                                  an_operator.the_species_extent_,
                                  an_operator.collision_interspecies_,
                                  an_operator.comb_mat_,
                                  an_operator.hat_As_,
                                  an_operator.hat_Zs_,
                                  an_operator.vparg_,
                                  an_operator.coeff_intdvpar_};

            for(OrdinalType the_block_first_phi = 0;
                the_block_first_phi < the_phi_extent;
                the_block_first_phi += Constant::kBlockExtentPhi) {
                for(OrdinalType the_block_first_theta = 0;
                    the_block_first_theta < the_theta_extent;
                    the_block_first_theta += Constant::kBlockExtentTheta) {
                    for(OrdinalType the_block_first_r = 0;
                        the_block_first_r < the_r_extent;
                        the_block_first_r += Constant::kBlockExtentR) {
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
