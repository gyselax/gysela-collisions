// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_OPERATOR_H
#define KOLIOP_COLLISION_OPERATOR_H

#include <Kokkos_Core.hpp>
#include <vector>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/cvcddriverscheduler.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Operator definition
        ////////////////////////////////////////////////////////////////////////

        /// TODO(Etienne M): INDEX We have a dilemma. CPU allows for large
        /// temporary, per thread, buffers. GPU do not (at least for LDS). Due
        /// to caches hierarchy, CPU like per thread, contiguous memory
        /// accesses. GPU do not because this kind of access leads to wastes in
        /// the memory transactions (notion of un-coalesced accesses). Using
        /// Kokkos, we hope to attain decent performance on both GPU and CPU.
        ///
        /// Assuming the following example, all arrays with layout left
        /// (Fortran).
        ///
        ///     collapse(3)
        ///     for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///         for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///             for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                 const Flo64 a = A(ir, itheta);
        ///                 const Flo64 b = B(ir, itheta, iphi);
        ///                 for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///                     const Flo64 c = C(ivpar);
        ///                     for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                         const Flo64 d = D(imu);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// Collapsing on the first 3 loops would lead to decent performance on
        /// CPU. On GPU though, this would be crappy as each thread would do
        /// linear access in C, D and E. Access in C and D are *ok* that bad,
        /// because, out of one memory transaction, conceptualize it as a cache
        /// line, only 8 bytes will be kept but the next fetch will be likely to
        /// hit the cache. Except that on MI250X cards, we have very few cache
        /// (8 Mio per GCD!). Access in E are bad because each thread gets a
        /// different ir value which means, thread n+1 will access memory in a
        /// stride of mumax*vparmax (strongly un-coalesced).
        ///
        /// We could try to inter-exchange the loops but this would then work
        /// badly on CPU.
        ///
        ///     collapse(3)
        ///     for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///         for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///             for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                 const Flo64 a = A(ir, itheta);
        ///                 const Flo64 c = C(ivpar);
        ///                 const Flo64 d = D(imu);
        ///                 for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///                     for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                         const Flo64 b = B(ir, itheta, iphi);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// In the end, the most versatile is to collapse everything, do
        /// redundant computation on GPU but dont waste what you bring from
        /// global memory and hope that the compiler will optimize the redundant
        /// loads and computations computation on CPU.
        ///
        ///     collapse(5)
        ///     for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///         for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///             for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                 for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///                     for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                         const Flo64 a = A(ir, itheta);
        ///                         const Flo64 c = C(ivpar);
        ///                         const Flo64 d = D(imu);
        ///                         const Flo64 b = B(ir, itheta, iphi);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        ///
        class Operator {
        public:
        protected:
        public:
            /// @brief
            /// @param mu_extent
            /// @param vpar_extent
            /// @param tor1_extent
            /// @param tor2_extent
            /// @param tor3_extent
            /// @param species_extent
            /// @param collision_interspecies
            /// @param SOL
            /// @param LIM
            /// @param ir_SOL_separatrix We do not need to shift it by -istart
            /// because we do not parallelize on ir across ranks.
            /// @param hat_As
            /// @param hat_Zs
            /// @param mug
            /// @param vparg
            /// @param coeff_intdmu
            /// @param coeff_intdvpar
            /// @param coeff_AD_r
            /// @param mask_buffer_r
            /// @param mask_LIM
            /// @param B_norm
            /// @param Bstar_s
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
            // Could be protected but we would have to use friend functions.
            // protected:

            //
            // Constant, externally computed values
            //

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

            //
            // Constant, internally computed values
            //

            koliop::utility::View::MDL3<ComputeFloatType> nustar_coeff_;

            //
            // Variable, internally computed values
            //

            koliop::utility::View::MDL4<ComputeFloatType> n_s_;
            koliop::utility::View::MDL4<ComputeFloatType> U_par_s_;
            koliop::utility::View::MDL4<ComputeFloatType> U_par_s_input_;
            koliop::utility::View::MDL4<ComputeFloatType> T_s_;
            koliop::utility::View::MDL5<ComputeFloatType> nu_ab_;
            koliop::utility::View::MDL4<ComputeFloatType> q_par_on_nT_s_;

            //
            // Miscellaneous tooling
            //

            std::vector<Constant::DeviceExecutionSpaceType> execution_space_pool_;

            //
            // Miscellaneous numerical tooling
            //

            CvCdDriverScheduler cvcd_drivers_scheduler_;

            koliop::utility::View::MDL<Flo64******> a_device_visible_f5d_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "a_device_visible_f5d_s_"),
                                                                            static_cast<std::size_t>(mu_extent_), static_cast<std::size_t>(vpar_extent_),
                                                                            static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_),
                                                                            static_cast<std::size_t>(species_extent_)};
        };

    } // namespace collision
} // namespace koliop

#endif
