// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_OPERATOR_H
#define KOLIOP_COLLISION_OPERATOR_H

#include <Kokkos_Core.hpp>
#include <vector>

#include "KOLIOP/collisions/cvcddriverscheduler.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collisions {

        // TODO(Etienne M): Into enums.
#define KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff  1
#define KOLIOP_EXTRACTION_IDENTIFIER_n_s           2
#define KOLIOP_EXTRACTION_IDENTIFIER_U_par_s       3
#define KOLIOP_EXTRACTION_IDENTIFIER_T_s           4
#define KOLIOP_EXTRACTION_IDENTIFIER_nu_ab         5
#define KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s 6

        ////////////////////////////////////////////////////////////////////////
        // Operator definition
        ////////////////////////////////////////////////////////////////////////

        class Operator {
        public:
        protected:
        public:
            Operator(uint64_t     mu_extent,
                     uint64_t     vpar_extent,
                     uint64_t     tor1_extent,
                     uint64_t     tor2_extent,
                     uint64_t     tor3_extent,
                     uint64_t     species_extent,
                     uint64_t     global_domain_r_extent,
                     uint64_t     global_domain_theta_extent,
                     uint64_t     local_domain_r_offset,
                     uint64_t     local_domain_theta_offset,
                     bool         collision_interspecies,
                     bool         SOL,
                     bool         LIM,
                     uint64_t     ir_SOL_separatrix,
                     const Flo64* hat_As,
                     const Flo64* hat_Zs,
                     const Flo64* mug,
                     const Flo64* vparg,
                     const Flo64* coeff_intdmu,
                     const Flo64* coeff_intdvpar,
                     const Flo64* coeff_AD_r,
                     const Flo64* mask_buffer_r,
                     const Flo64* mask_LIM,
                     const Flo64* B_norm,
                     const Flo64* Bstar_s);

            ~Operator();

            void Fence(OrdinalType execution_space_index) const;

            void FenceAll() const;

            void ComputeMoments(Flo64* f5d_s);

            void PrecomputeNu();

            void ComputeMaxwellianEvolution(ComputeFloatType dt,
                                            Flo64*           f5d_s);

            void ComputeQparOnNT(const Flo64* f5d_s);

            void ComputeCpar(ComputeFloatType dt,
                             Flo64*           f5d_s) const;

            void ComputeMaxwellianUnshiftedLoc(uint32_t choice,
                                               Flo64*   f5d_s) const;

            void Apply(ComputeFloatType dt,
                       Flo64*           f5d_s);

            void ComputeDensityCorrection(Flo64* f5d_s) const;

            void ComputeUparTCorrection(ComputeFloatType dt,
                                        Flo64*           f5d_s);

            void ExtractRaw(uint32_t value_identifier,
                            void*    an_output_buffer) const;

            /// Compute the evolution of the distribution function due to
            /// collisions. The collision are applied to all the species in the
            /// [0, species_extent_) range.
            ///
            void Collision(ComputeFloatType dt,
                           Flo64*           f5d_s);

        protected:
            /// Compute the evolution of the distribution function due to
            /// collisions. The collision are applied to all the species in the
            /// [first_species, last_species) range.
            ///
            template <typename ForwardIterator>
            void CollisionFor(ComputeFloatType dt,
                              Flo64*           f5d_s,
                              ForwardIterator  first_species,
                              ForwardIterator  last_species);

        public:
            // TODO(Etienne M): protected:

            // Constant, externally computed values
            OrdinalType mu_extent_;
            OrdinalType vpar_extent_;
            OrdinalType tor1_extent_;
            OrdinalType tor2_extent_;
            OrdinalType tor3_extent_;
            OrdinalType r_extent_;
            OrdinalType theta_extent_;
            OrdinalType phi_extent_;
            OrdinalType species_extent_;
            OrdinalType global_domain_r_extent_;
            OrdinalType global_domain_theta_extent_;
            OrdinalType local_domain_r_offset_;
            OrdinalType local_domain_theta_offset_;
            bool        collision_interspecies_;
            bool        SOL_;
            bool        LIM_;
            OrdinalType ir_SOL_separatrix_;

            // TODO(Etienne M): we store Flo64 but it should be
            // ComputeFloatType. To achieve this, we would need to handle
            // conversions from Flo64 to whatever ComputeFloatType is (say
            // Flo32).

            koliop::utility::View::InternalInputL1<const Flo64> hat_As_;
            koliop::utility::View::InternalInputL1<const Flo64> hat_Zs_;
            koliop::utility::View::InternalInputL1<const Flo64> mug_;
            koliop::utility::View::InternalInputL1<const Flo64> vparg_;
            koliop::utility::View::InternalInputL1<const Flo64> coeff_intdmu_;
            koliop::utility::View::InternalInputL1<const Flo64> coeff_intdvpar_;
            koliop::utility::View::InternalInputL1<const Flo64> coeff_AD_r_;
            koliop::utility::View::InternalInputL1<const Flo64> mask_buffer_r_;
            koliop::utility::View::InternalInputL2<const Flo64> mask_LIM_;
            koliop::utility::View::InternalInputL2<const Flo64> B_norm_;
            koliop::utility::View::InternalInputL4<const Flo64> Bstar_s_;

            // Constant, internally computed values
            koliop::utility::View::MDL3<ComputeFloatType> nustar_coeff_;

            // Variable, internally computed values
            koliop::utility::View::MDL4<ComputeFloatType> n_s_;
            koliop::utility::View::MDL4<ComputeFloatType> U_par_s_;
            koliop::utility::View::MDL4<ComputeFloatType> U_par_s_input_;
            koliop::utility::View::MDL4<ComputeFloatType> T_s_;
            koliop::utility::View::MDL5<ComputeFloatType> nu_ab_;
            koliop::utility::View::MDL4<ComputeFloatType> q_par_on_nT_s_;

            // Miscellaneous tooling
            std::vector<Constant::DeviceExecutionSpaceType> execution_space_pool_;

            // Miscellaneous numerical tooling
            CvCdDriverScheduler cvcd_drivers_scheduler_;
        };

    } // namespace collisions
} // namespace koliop

#endif
