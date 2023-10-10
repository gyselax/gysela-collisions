// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/cvcddriverscheduler.h"

#include "KOLIOP/collision/cvcddriver.h"
#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {
        namespace detail {

            static inline void
            PrepareBlockPosition(OrdinalType                                                the_local_domain_r_extent,
                                 OrdinalType                                                the_local_domain_theta_extent,
                                 OrdinalType                                                the_local_domain_phi_extent,
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


                        // If we could create some block, decrease the local
                        // domain by as much as we have fitted of.
                        the_local_domain_r_extent -= the_block_count_in_r * the_current_r_block_extent;
                        the_local_domain_theta_extent -= the_block_count_in_theta * the_current_theta_block_extent;
                        the_local_domain_phi_extent -= the_block_count_in_phi * the_current_phi_block_extent;
                    }

                    // NOTE: We could early exit if at least
                    // the_local_domain_XXX_extent == 0.

                    // NOTE: The block wont fit, go to the next (smaller) block
                    // size.
                }

                KOLIOP_ASSERT(the_local_domain_r_extent == 0 &&
                              the_local_domain_theta_extent == 0 &&
                              the_local_domain_phi_extent == 0);
            }

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

            detail::PrepareBlockPosition(an_operator.the_r_extent_,
                                         an_operator.the_theta_extent_,
                                         an_operator.the_phi_extent_,
                                         the_cvcd_drivers_configurations_,
                                         the_preset_block_extent_count,
                                         the_preset_block_extent);

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            for(auto&& a_driver_configuration : the_cvcd_drivers_configurations_) {
                detail::CreateCvCdDriver(an_operator,
                                         a_driver_configuration.the_cvcd_driver_,
                                         a_driver_configuration.the_block_r_extent_,
                                         a_driver_configuration.the_block_theta_extent_,
                                         a_driver_configuration.the_block_phi_extent_);

                // the_cvcd_drivers_configurations_.size() % the_execution_space_pool.size()
                a_driver_configuration.the_execution_space_ = the_execution_space_pool[0];
            }

            // NOTE: we have an issue with the amount of memory allocated !!
            // NOTE: use a shared ptr in the eraser ? reuse the same cvcddrier
            // n times ?
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
