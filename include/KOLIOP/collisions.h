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

        ////////////////////////////////////////////////////////////////////////
        // Operator definition
        ////////////////////////////////////////////////////////////////////////

        class Operator {
        public:
            enum class Diagnostics {
                k_nustar_coeff  = 1,
                k_n_s           = 2,
                k_U_par_s       = 3,
                k_T_s           = 4,
                k_nu_ab         = 5,
                k_q_par_on_nT_s = 6,
            };

        protected:
        public:
            /// @brief Instantiate an operator state. All the input are expected in layout left. Meaning an_array(i,j,k)
            /// is expected to be iterated quickly on the left most dimension. Equivalently, an_array(k,j,i) in layout
            /// right will work fine. We take a view (reference) of the input arrays, you may not deallocate them while
            /// the operator is not released. The values given below are expected NOT to change during the lifetime of
            /// the operator.
            /// @param tor1_extent The rank's theta domain extent in grid point.
            /// @param tor2_extent The rank's r domain extent in grid point.
            /// @param tor3_extent The rank's phi domain extent in grid point.
            /// @param collision_interspecies Boolean as type int8_t (0 = false, else true) describing if a species
            /// collision with itself.
            /// @param SOL Boolean as type int8_t (0 = false, else true) used in the Nu* computation.
            /// @param LIM Boolean as type int8_t (0 = false, else true) set to true to tell the operator to apply a
            /// limiter mask.
            /// @param ir_SOL_separatrix Fix the collisionality past ir_SOL_separatrix to the value at
            /// ir_SOL_separatrix. Must respect the following constraint: 0 <= ir_SOL_separatrix < r. In Gysela fortran,
            /// ir_SOL_separatrix must respect the following constraint: 0 <= ir_SOL_separatrix <= r where MAX_R_START.
            /// Used in the Nu computation. Used for a specific treatment of the SOL.
            /// @param hat_As An array of double with extents ([0, species_extent) ).
            /// @param hat_Zs An array of double with extents ([0, species_extent) ).
            /// @param mug An array of double with extents ([0, mu_extent) ) representing the coordinate of the Mu grid
            /// points.
            /// @param vparg An array of double with extents ([0, vpar_extent) ) representing the coordinate of the Vpar
            /// grid points.
            /// @param coeff_intdmu An array of double with extents ([0, mu_extent) ) representing the coefficients for
            /// the integral of the coordinates in Mu.
            /// @param coeff_intdvpar An array of double with extents ([0, vpar_extent) ) representing the coefficients
            /// for the integral of the coordinates in Vpar (generally using the simpson).
            /// @param coeff_AD_r An array of double with extents ([0, r) ) representing the AD coefficients.
            /// @param mask_buffer_r An array of double with extents ([0, r) ) representing an operator mask in R.
            /// @param mask_LIM An array of double with extents ([0, r), [0, theta) ) representing an operator's limiter
            /// mask. Must be set if LIM is true else, it may be a NULL.
            /// @param B_norm An array of double with extents ([0, r), [0, theta) ).
            /// @param Bstar_s An array of double with extents ([0, vpar_extent), [0, r), [0, theta), [0,
            /// species_extent) ), note that we do not use torN_extent as these may change depending on library calling
            /// Koliop.
            Operator(uint64_t                                        tor1_extent,
                     uint64_t                                        tor2_extent,
                     uint64_t                                        tor3_extent,
                     bool                                            collision_interspecies,
                     bool                                            SOL,
                     bool                                            LIM,
                     uint64_t                                        ir_SOL_separatrix,
                     const koliop::utility::View::MDL1<const Flo64>& hat_As,
                     const koliop::utility::View::MDL1<const Flo64>& hat_Zs,
                     const koliop::utility::View::MDL1<const Flo64>& mug,
                     const koliop::utility::View::MDL1<const Flo64>& vparg,
                     const koliop::utility::View::MDL1<const Flo64>& coeff_intdmu,
                     const koliop::utility::View::MDL1<const Flo64>& coeff_intdvpar,
                     const koliop::utility::View::MDL1<const Flo64>& coeff_AD_r,
                     const koliop::utility::View::MDL1<const Flo64>& mask_buffer_r,
                     const koliop::utility::View::MDL2<const Flo64>& mask_LIM,
                     const koliop::utility::View::MDL2<const Flo64>& B_norm,
                     const koliop::utility::View::MDL4<const Flo64>& Bstar_s);

            ~Operator();

            void Fence(OrdinalType execution_space_index) const;

            /// @brief Ensure all the computation the user asked of the operator are visible in memory.
            void FenceAll() const;

            /// @brief Computes the fluid moments:
            ///   na =  int 2 * Pi * Bstar * dmu * dvpar * Fa
            ///   Vpara = (1/na) * int 2 * Pi * Bstar * dmu * dvpar * vpar * Fa
            ///   Ta = (1/(3*na)) * int 2 * Pi * Bstar * dmu * dvpar * Fa * &
            ///        (2 * mu *B + (vpar-Vpara)^2)
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void ComputeMoments(const koliop::utility::View::MDL6<Flo64>& f5d_s);

            /// @brief Collisionality
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            void PrecomputeNu();

            /// @brief Make the Maxwellian evolve:
            ///     - remove the Maxwellian to the distribution function Fa = Fa - FMa;
            ///     - evolve temperature;
            ///     - compute the new Maxwellian;
            ///     - add it to the function distribution Fa = Fa + FMa.
            /// Evolution of the temperatures via the term C0:
            ///   Ta = Ta + deltat*sum_b nu_ab * (2*hat_Aa/(hat_Aa + hat_Ab)*(Tb -Ta) +
            ///     (2/3) * Vpara *(Vpara - Vparb))
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param dt The collision dt (it may be larger than the simulation's dt).
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void ComputeMaxwellianEvolution(ComputeFloatType                          dt,
                                            const koliop::utility::View::MDL6<Flo64>& f5d_s);

            /// @brief Computation of q_para/(na*Ta):
            ///   q_par_on_nT = 0.5*int (2 * Pi * Bstar) *  dmu * dvpar * &
            ///       (vpar-Vpara) * ((vpar-Vpara)^2 + vperp^2) * (Fa - FMa)
            ///  Indeed  q_par_on_nT = 0 for a Maxwellian distribution function
            ///
            /// WARNING: Either call Fence() after this call or chain with other
            /// koliop_* operator's operations. You still may have to fence after the last
            /// operation.
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void ComputeQparOnNT(const koliop::utility::View::MDL6<const Flo64>& f5d_s);

            /// @brief Make Fa evolve:
            ///   f = f + dt * sum_b (nu_sab(xb) * vpar / Ta) * F_M0a * &
            ///           (Rat * Vparb - Vpara + (q_para / (na * Ta)) * (1 - 2*xa*xa/5) -
            ///           (3/5) *  Rat * (q_parb / (nb * Tb)) / (1 + (vTa/vTb)^2 ))
            ///  with Rat = sqrt(hat_Aa/hat_Ab)
            ///   xa = v /(sqrt(2*Ta)) = sqrt(2*mu*B + vpar*vpar)/(sqrt(2*Ta))
            ///   xb = xa * vTa / vTb
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param dt The collision dt (it may be larger than the simulation's dt).
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void ComputeCpar(ComputeFloatType                          dt,
                             const koliop::utility::View::MDL6<Flo64>& f5d_s) const;

            /// @brief This routine compute one of the following operation on the
            /// distribution function:
            ///  - choice 3 : Fa = Fa * FM0a
            ///  - choice 4 : Fa = Fa / FM0a
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            /// @param choice [3, 4]
            void ComputeMaxwellianUnshiftedLoc(uint32_t                                  choice,
                                               const koliop::utility::View::MDL6<Flo64>& f5d_s) const;

            /// @brief Compute alpha, make alpha evolve and reconstruct Fa.
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param dt The collision dt (it may be larger than the simulation's dt).
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void Apply(ComputeFloatType                          dt,
                       const koliop::utility::View::MDL6<Flo64>& f5d_s);

            /// @brief Correction to keep the local density constant. After the main part of the collisions, the
            /// distribution function has evolved:
            ///   C(Fa) = Fa'
            /// The new density na' can be different than the one before collisions In order to keep the local density
            /// constant one can applied a correction after collisions by taking Fa'' = Fa' * na /na'
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void ComputeDensityCorrection(const koliop::utility::View::MDL6<Flo64>& f5d_s) const;

            /// @brief Correction to keep the local fluid moments constant. init corresponds to values before
            /// collisions. old corresponds to values after collisions but before correction. new corresponds to
            /// corrected values.
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param dt The collision dt (it may be larger than the simulation's dt).
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void ComputeUparTCorrection(ComputeFloatType                          dt,
                                        const koliop::utility::View::MDL6<Flo64>& f5d_s);

            /// @brief Extract an internal buffer into output_buffer. The buffer in question is specified using
            /// value_identifier and the operator::Diagnostics enumerations.
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param value_identifier A Diagnostics element.
            /// @param output_buffer A correctly sized buffer.
            void ExtractRaw(Diagnostics value_identifier,
                            void*       an_output_buffer) const;

            /// @brief Computes the evolution of the distribution function due to collisions. It computes the collision
            /// of all species against all other species (if asked for through collision_interspecies and
            /// species_extent). This "do it all method" is used by Gyselalib++.
            /// The operation, in order, are:
            /// - Compute fluid moments n, Vpar, T, q_par (koliop_ComputeMoments);
            /// - Make the Maxwellian evolve (koliop_ComputeMaxwellianEvolution);
            /// - Compute the normalized function distribution Fa = Fa / FM0a (koliop_ComputeMaxwellianUnshiftedLoc);
            /// - Begin loops on r, theta, phi:
            ///     - Make Fa evolve via collvparmu_general_Cpar (koliop_ComputeCpar);
            ///     - Compute alpha via collvparmu_general_alphaprojection (koliop_Apply);
            ///     - Make alpha evolve via collvparmu_general_C0 (koliop_Apply);
            ///     - Make alpha evolve via collvparmu_general_CvCd (koliop_Apply);
            ///     - Make alpha evolve via collvparmu_general_C0 (koliop_Apply);
            ///     - Reconstruct Fa via collvparmu_general_reconstruction (koliop_Apply);
            ///     - Make Fa evolve via collvparmu_general_Cpar (koliop_ComputeCpar);
            /// - End loops
            /// - Compute the real function distribution Fa = Fa * FM0a (koliop_ComputeMaxwellianUnshiftedLoc);
            /// - Make the Maxwellian evolve via collvparmu_evolution_Maxwellian (koliop_ComputeMaxwellianEvolution).
            ///
            /// WARNING: Either call Fence() after this call or chain with other koliop_* operator's operations. You
            /// still may have to fence after the last operation.
            /// @param dt The collision dt (it may be larger than the simulation's dt).
            /// @param f5d_s An array of double with extents ([0, mu_extent), [0, vpar_extent), [0, tor1_extent), [0,
            /// tor2_extent), [0, tor3_extent), [0, species_extent)).
            void Collisions(ComputeFloatType                          dt,
                            const koliop::utility::View::MDL6<Flo64>& f5d_s);

        public:
            // Constant, externally computed values
            OrdinalType mu_extent_;
            OrdinalType vpar_extent_;
            OrdinalType tor1_extent_;
            OrdinalType tor2_extent_;
            OrdinalType tor3_extent_;
            OrdinalType species_extent_;
            bool        collision_interspecies_;
            bool        SOL_;
            bool        LIM_;
            OrdinalType ir_SOL_separatrix_;

            koliop::utility::View::MDL1<const Flo64> hat_As_;
            koliop::utility::View::MDL1<const Flo64> hat_Zs_;
            koliop::utility::View::MDL1<const Flo64> mug_;
            koliop::utility::View::MDL1<const Flo64> vparg_;
            koliop::utility::View::MDL1<const Flo64> coeff_intdmu_;
            koliop::utility::View::MDL1<const Flo64> coeff_intdvpar_;
            koliop::utility::View::MDL1<const Flo64> coeff_AD_r_;
            koliop::utility::View::MDL1<const Flo64> mask_buffer_r_;
            koliop::utility::View::MDL2<const Flo64> mask_LIM_;
            koliop::utility::View::MDL2<const Flo64> B_norm_;
            koliop::utility::View::MDL4<const Flo64> Bstar_s_;

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
