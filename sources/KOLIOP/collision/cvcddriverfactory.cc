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
                                  OrdinalType       block_tor1_extent,
                                  OrdinalType       block_tor2_extent,
                                  OrdinalType       block_tor3_extent) {

                const auto& vpar_extent    = an_operator.vpar_extent_;
                const auto& species_extent = an_operator.species_extent_;

                const auto& hat_As         = an_operator.hat_As_;
                const auto& hat_Zs         = an_operator.hat_Zs_;
                const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
                const auto& vparg          = an_operator.vparg_;

                const auto& collision_interspecies = an_operator.collision_interspecies_;

                // TODO(Etienne M): Build a constexpr thingy that builds the
                // factory out of Constant::kPresetBlockExtent

                if(false) {
                } else if(block_tor1_extent == 256 && block_tor2_extent == 128 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<256, 128, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 256 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 256, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 256 && block_tor2_extent == 128 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<256, 128, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 256 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 256, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 128 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 128, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 64 && block_tor3_extent == 4) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 4, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 128 && block_tor3_extent == 4) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 4, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 32 && block_tor3_extent == 8) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 8, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 128 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 128, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 64 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 32 && block_tor3_extent == 4) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 4, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 128 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 64 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 64, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 32 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 128 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 128, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 64 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 64, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 32 && block_tor2_extent == 128 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 128, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 128 && block_tor2_extent == 32 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<128, 32, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 64 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 64, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 32 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<64, 32, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 32 && block_tor2_extent == 128 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 128, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 32 && block_tor2_extent == 64 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 64, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 64 && block_tor2_extent == 32 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<64, 32, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 32 && block_tor2_extent == 64 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 64, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 32 && block_tor2_extent == 32 && block_tor3_extent == 2) {
                    a_cvcd_driver.Set(CvCdDriver<32, 32, 2, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 32 && block_tor2_extent == 32 && block_tor3_extent == 1) {
                    a_cvcd_driver.Set(CvCdDriver<32, 32, 1, Constant::BatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else if(block_tor1_extent == 1 && block_tor2_extent == 1 && block_tor3_extent == 1) {
                    // NOTE: the FakeBatchTraitsType is important.
                    a_cvcd_driver.Set(CvCdDriver<1, 1, 1, Constant::FakeBatchTraitsType>{vpar_extent, species_extent, collision_interspecies, hat_As, hat_Zs, vparg, coeff_intdvpar});
                } else {
                    // The block size is registered in
                    // preset_block_extent but we are not able to build
                    // such block of this size.
                    KOLIOP_ASSERT(false && "For some reason, the CvCd operator blocking configuration you choose is not part of the factory.");
                }
            }

        } // namespace detail
    }     // namespace collision
} // namespace koliop
