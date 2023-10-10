// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/cvcddriverscheduler.h"

#include "KOLIOP/collision/cvcddriver.h"
#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {
        namespace detail {

            // Sort of factory.
            static inline void
            CreateCvCdDriver(const Operator&   an_operator,
                             CvCdDriverEraser& a_cvcd_driver,
                             OrdinalType       the_block_first_r,
                             OrdinalType       the_block_first_theta,
                             OrdinalType       the_block_first_phi) {

                const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
                const auto& the_species_extent = an_operator.the_species_extent_;

                const auto& comb_mat       = an_operator.comb_mat_;
                const auto& hat_As         = an_operator.hat_As_;
                const auto& hat_Zs         = an_operator.hat_Zs_;
                const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
                const auto& vparg          = an_operator.vparg_;

                const auto& pSpecies_id            = an_operator.pSpecies_id_;
                const auto& collision_interspecies = an_operator.collision_interspecies_;

                // TODO(Etienne M): Build a constexpr thingy that builds the
                // factory out of Constant::kPresetBlockExtent

                if(the_block_first_r == 256 && the_block_first_theta == 128 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<256, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 256 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 256, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 256 && the_block_first_theta == 128 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<256, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 256 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 256, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 128 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 128 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 64 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 64 && the_block_first_theta == 128 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 64 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 32 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 64 && the_block_first_theta == 128 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 64 && the_block_first_theta == 64 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 64, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 32 && the_block_first_theta == 128 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 128 && the_block_first_theta == 32 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 64 && the_block_first_theta == 64 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 64, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 64 && the_block_first_theta == 32 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 32, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 32 && the_block_first_theta == 128 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 32 && the_block_first_theta == 64 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 64, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 64 && the_block_first_theta == 32 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 32, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 32 && the_block_first_theta == 64 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 64, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 32 && the_block_first_theta == 32 && the_block_first_phi == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 32, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_first_r == 32 && the_block_first_theta == 32 && the_block_first_phi == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 32, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else {
                    // The block size is registered in
                    // the_preset_block_extent but we are not able to build
                    // such block of this size.
                    KOLIOP_ASSERT(false);
                }
            }

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

            // TODO(Etienne M): Blocking idea, use a recursive approach where we
            // try to fit the biggest block size to process the domain, and if
            // not possible we fallback to a smaller size. Then, we process the
            // potential remaining grid point recursively. Supposing we do not
            // template CvCd on the block size anymore, we could reuse them for
            // smaller blocs.

            // in the constants, create a static cosnexpr array of the size we wish to support. when initialization the operator, depending on the size of the localc domain, instantiate the gbest set of cvcd driver.
            // also, register the sequence of cvcd driver and the position they should execute at in an array.

            // Basicaly, in the cosntructor of the operator, buidl a table of indices:

            // NOTE: do not forget to add a loggin optio to get the plannified schedule.

            auto the_local_domain_r_extent     = an_operator.the_r_extent_;
            auto the_local_domain_theta_extent = an_operator.the_theta_extent_;
            auto the_local_domain_phi_extent   = an_operator.the_phi_extent_;

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

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

                while(the_local_domain_r_extent >= the_current_r_block_extent &&
                      the_local_domain_theta_extent >= the_current_theta_block_extent &&
                      the_local_domain_phi_extent >= the_current_phi_block_extent) {
                    // NOTE: While we can breakdown the domain in block of
                    // the_current_block_extent

                    the_local_domain_r_extent -= the_current_r_block_extent;
                    the_local_domain_theta_extent -= the_current_theta_block_extent;
                    the_local_domain_phi_extent -= the_current_phi_block_extent;

                    auto& a_driver_configuration = the_cvcd_drivers_configurations_.emplace_back();

                    detail::CreateCvCdDriver(an_operator,
                                             a_driver_configuration.the_cvcd_driver_,
                                             the_current_r_block_extent,
                                             the_current_theta_block_extent,
                                             the_current_phi_block_extent);

                    // NOTE: it is as if we go backward to fill the local
                    // domain. For instance, a 1D domain of size 64:
                    // A initialization the space that has not yet been
                    // associated to a block (call it s), is 64. Say we have
                    // blocks of size 8 and 16.
                    // |----------------| s = 64
                    // |------------++++| s >= 16 -> s = s - 16 -> 48. This block will handle range [48, 64)
                    // |--------++++++++| s >= 16 -> s = s - 16 -> 32. This block will handle range [32, 48)
                    // etc.
                    a_driver_configuration.the_block_first_r_     = the_local_domain_r_extent;
                    a_driver_configuration.the_block_first_theta_ = the_local_domain_theta_extent;
                    a_driver_configuration.the_block_first_phi_   = the_local_domain_phi_extent;

                    a_driver_configuration.the_execution_space_ = the_execution_space_pool[the_cvcd_drivers_configurations_.size() %
                                                                                           the_execution_space_pool.size()];
                }

                // NOTE: The block wont fit, go to the next (smaller) block
                // size.
            }

            KOLIOP_ASSERT(the_local_domain_r_extent == 0 &&
                          the_local_domain_theta_extent == 0 &&
                          the_local_domain_phi_extent == 0);

            // NOTE: we have an issue with the amount of memory allocated !!
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
