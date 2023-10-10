// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/cvcddriver.h"
#include "KOLIOP/collision/cvcddrivereraser.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace collision {
        namespace detail {

            void CreateCvCdDriver(const Operator&   an_operator,
                                  CvCdDriverEraser& a_cvcd_driver,
                                  OrdinalType       the_block_r_extent,
                                  OrdinalType       the_block_theta_extent,
                                  OrdinalType       the_block_phi_extent) {

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

                if(the_block_r_extent == 256 && the_block_theta_extent == 128 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<256, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 256 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 256, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 256 && the_block_theta_extent == 128 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<256, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 256 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 256, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 128 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 128 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 64 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 64 && the_block_theta_extent == 128 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 64 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 32 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 64 && the_block_theta_extent == 128 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 64 && the_block_theta_extent == 64 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 64, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 32 && the_block_theta_extent == 128 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 128, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 128 && the_block_theta_extent == 32 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 64 && the_block_theta_extent == 64 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 64, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 64 && the_block_theta_extent == 32 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 32, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 32 && the_block_theta_extent == 128 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 128, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 32 && the_block_theta_extent == 64 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 64, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 64 && the_block_theta_extent == 32 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 32, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 32 && the_block_theta_extent == 64 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 64, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 32 && the_block_theta_extent == 32 && the_block_phi_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 32, 2, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(the_block_r_extent == 32 && the_block_theta_extent == 32 && the_block_phi_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 32, 1, Constant::BatchTraitsType>{pSpecies_id, the_vpar_extent, the_species_extent, collision_interspecies, comb_mat, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else {
                    // The block size is registered in
                    // the_preset_block_extent but we are not able to build
                    // such block of this size.
                    KOLIOP_ASSERT(false);
                }
            }

        } // namespace detail
    }     // namespace collision
} // namespace koliop
