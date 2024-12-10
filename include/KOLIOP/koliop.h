// SPDX-License-Identifier: MIT

#ifndef KOLIOP_KOLIOP_H
#define KOLIOP_KOLIOP_H

// NOTE: Everything in this file should be C99.

#include <stddef.h>
#include <stdint.h>

#include "KOLIOP/error/status.h"
#include "KOLIOP/internal/buildspecificities.h"


////////////////////////////////////////////////////////////////////////////////
// koliop constants declaration
////////////////////////////////////////////////////////////////////////////////

#define KOLIOP_FALSE 0
#define KOLIOP_TRUE  1


////////////////////////////////////////////////////////////////////////////////
// koliop method declaration
////////////////////////////////////////////////////////////////////////////////

#if defined(__cplusplus)
extern "C" {
#endif

/// Has to be called before any other koliop function is called.
/// @param a_host_thread_count The host threads to use to drive the device or to
/// do the computation. Depending on how koliop was built.
/// NOTE: Under the hood, we call ::omp_set_num_threads(a_host_thread_count). So
/// make sure you pass a correct value.
/// @param a_device_index The device koliop will use to offload the work. The
/// value as no meaning if koliop was not built with device support enabled.
/// @param with_verbose_initialization KOLIOP_FALSE to disable or KOLIOP_TRUE to
/// present additional information.
/// @return KOLIOP_STATUS_SUCCESS on success.
///
koliop_Status koliop_Initialize(int8_t do_kokkos_initialization,
                                size_t a_host_thread_count,
                                size_t a_device_index,
                                int8_t with_verbose_initialization);

/// Has to be called when koliop is not to be used anymore for the
/// lifetime of the program.
/// @return KOLIOP_STATUS_SUCCESS on success.
///
koliop_Status koliop_Deinitialize(int8_t do_kokkos_deinitialization);

/// Represent a collision operator.
///
typedef struct koliop_OpaqueOperator_* koliop_Operator;

/// @brief Instantiate an operator state. All the input are expected in layout
/// left. Meaning an_array(i,j,k) is expected to be iterated quickly on the left
/// most dimension. Equivalently, an_array(k,j,i) in layout right will work
/// fine. We take a view (reference) of the input arrays, you may not deallocate
/// them while the operator is not released.
/// The values given below are expected NOT to change during the lifetime
/// of the operator.
/// @param an_operator_handle A pointer to a koliop_Operator. This handle represents the operator as an opaque pointer.
/// @param the_mu_extent The rank's Mu domain extent in grid point.
/// @param the_vpar_extent The rank's Vpar domain extent in grid point.
/// @param the_r_extent The rank's R domain extent in grid point.
/// @param the_theta_extent The rank's Theta domain extent in grid point.
/// @param the_phi_extent The rank's Phi domain extent in grid point.
/// @param the_species_extent The rank's Species domain extent (aka number of species).
/// @param the_global_domain_r_extent The simulation's R domain extent in grid point.
/// @param the_global_domain_theta_extent The simulation's Theta domain extent in grid point.
/// @param the_local_domain_r_offset The rank's R offset in the simulation's domain.
/// @param the_local_domain_theta_offset The rank's R offset in the simulation's domain.
/// @param collision_interspecies Boolean as type int8_t (0 = false, else true) describing if a species collision with itself.
/// @param SOL Boolean as type int8_t (0 = false, else true) used in the Nu* computation.
/// @param LIM Boolean as type int8_t (0 = false, else true) set to true to tell the operator to apply a limiter mask.
/// @param ir_SOL_separatrix Fix the collisionality past ir_SOL_separatrix to the value at ir_SOL_separatrix. Must respect the following constraint: 0 <= ir_SOL_separatrix < the_global_domain_r_extent. In Gysela fortran, ir_SOL_separatrix must respect the following constraint: MAX_R_START <= ir_SOL_separatrix <= the_global_domain_r_extent where MAX_R_START is the the_local_domain_r_offset of a rank processing the most outer subdomain in R. Used in the Nu computation. Used for a specific treatment of the SOL.
/// @param hat_As An array of double with extents ([0, the_species_extent) ).
/// @param hat_Zs An array of double with extents ([0, the_species_extent) ).
/// @param mug An array of double with extents ([0, the_mu_extent) ) representing the coordinate of the Mu grid points.
/// @param vparg An array of double with extents ([0, the_vpar_extent) ) representing the coordinate of the Vpar grid points.
/// @param coeff_intdmu An array of double with extents ([0, the_mu_extent) ) representing the coefficients for the integral of the coordinates in Mu.
/// @param coeff_intdvpar An array of double with extents ([0, the_vpar_extent) ) representing the coefficients for the integral of the coordinates in Vpar (generally using the simpson).
/// @param coeff_AD_r An array of double with extents ([0, the_r_extent) ) representing the AD coefficients.
/// @param mask_buffer_r An array of double with extents ([0, the_global_domain_r_extent) ) representing an operator mask in R.
/// @param mask_LIM An array of double with extents ([0, the_global_domain_r_extent), [0, the_global_domain_theta_extent) ) representing an operator's limiter mask. Must be set if LIM is true else, it may be a NULL.
/// @param B_norm An array of double with extents ([0, the_global_domain_r_extent), [0, the_global_domain_theta_extent) ).
/// @param Bstar_s An array of double with extents ([0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
                            size_t           the_mu_extent,
                            size_t           the_vpar_extent,
                            size_t           the_r_extent,
                            size_t           the_theta_extent,
                            size_t           the_phi_extent,
                            size_t           the_species_extent,
                            size_t           the_global_domain_r_extent,
                            size_t           the_global_domain_theta_extent,
                            size_t           the_local_domain_r_offset,
                            size_t           the_local_domain_theta_offset,
                            int8_t           collision_interspecies,
                            int8_t           SOL,
                            int8_t           LIM,
                            int64_t          ir_SOL_separatrix,
                            const double*    hat_As,
                            const double*    hat_Zs,
                            const double*    mug,
                            const double*    vparg,
                            const double*    coeff_intdmu,
                            const double*    coeff_intdvpar,
                            const double*    coeff_AD_r,
                            const double*    mask_buffer_r,
                            const double*    mask_LIM,
                            const double*    B_norm,
                            const double*    Bstar_s);

/// @brief Does an implied koliop_Fence and release the memory associated to the
/// operator instance.
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_Release(koliop_Operator an_operator);

/// @brief Ensure all the computation the user asked of the operator are visible
/// in memory.
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_Fence(koliop_Operator an_operator);

/// @brief Computes the fluid moments:
///   na =  int 2 * Pi * Bstar * dmu * dvpar * Fa
///   Vpara = (1/na) * int 2 * Pi * Bstar * dmu * dvpar * vpar * Fa
///   Ta = (1/(3*na)) * int 2 * Pi * Bstar * dmu * dvpar * Fa * &
///        (2 * mu *B + (vpar-Vpara)^2)
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeMoments(koliop_Operator an_operator,
                                    double*         f5d_s);

/// @brief Collisionality
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_PrecomputeNu(koliop_Operator an_operator);

/// @brief Make the Maxwellian evolve:
///     - remove the Maxwellian to the distribution function Fa = Fa - FMa;
///     - evolve temperature;
///     - compute the new Maxwellian;
///     - add it to the function distribution Fa = Fa + FMa.
/// Evolution of the temperatures via the term C0:
///   Ta = Ta + deltat*sum_b nu_ab * (2*hat_Aa/(hat_Aa + hat_Ab)*(Tb -Ta) +
///     (2/3) * Vpara *(Vpara - Vparb))
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param dt The collision dt (it may be larger than the simulation's dt).
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeMaxwellianEvolution(koliop_Operator an_operator,
                                                double          dt,
                                                double*         f5d_s);

/// @brief Computation of q_para/(na*Ta):
///   q_par_on_nT = 0.5*int (2 * Pi * Bstar) *  dmu * dvpar * &
///       (vpar-Vpara) * ((vpar-Vpara)^2 + vperp^2) * (Fa - FMa)
///  Indeed  q_par_on_nT = 0 for a Maxwellian distribution function
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeQparOnNT(koliop_Operator an_operator,
                                     const double*   f5d_s);

/// @brief Make Fa evolve:
///   f = f + dt * sum_b (nu_sab(xb) * vpar / Ta) * F_M0a * &
///           (Rat * Vparb - Vpara + (q_para / (na * Ta)) * (1 - 2*xa*xa/5) -
///           (3/5) *  Rat * (q_parb / (nb * Tb)) / (1 + (vTa/vTb)^2 ))
///  with Rat = sqrt(hat_Aa/hat_Ab)
///   xa = v /(sqrt(2*Ta)) = sqrt(2*mu*B + vpar*vpar)/(sqrt(2*Ta))
///   xb = xa * vTa / vTb
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param dt The collision dt (it may be larger than the simulation's dt).
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeCpar(koliop_Operator an_operator,
                                 double          dt,
                                 double*         f5d_s);

/// @brief This routine compute one of the following operation on the
/// distribution function:
///  - choice 3 : Fa = Fa * FM0a
///  - choice 4 : Fa = Fa / FM0a
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @param choice [3, 4]
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeMaxwellianUnshiftedLoc(koliop_Operator an_operator,
                                                   double*         f5d_s,
                                                   int32_t         choice);

/// @brief Compute alpha, make alpha evolve and reconstruct Fa.
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param dt The collision dt (it may be larger than the simulation's dt).
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_Apply(koliop_Operator an_operator,
                           double          dt,
                           double*         f5d_s);

/// @brief Correction to keep the local density constant
/// After the main part of the collisions, the distribution function has
///  evolved:
///   C(Fa) = Fa'
/// The new density na' can be different than the one before collisions
/// In order to keep the local density constant one can applied a correction
///  after collisions by taking Fa'' = Fa' * na /na'
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeDensityCorrection(koliop_Operator an_operator,
                                              double*         f5d_s);

/// @brief Correction to keep the local fluid moments constant.
/// init corresponds to values before collisions.
/// old corresponds to values after collisions but before correction.
/// new corresponds to corrected values.
///
/// WARNING: Either call koliop_Fence after this call or chain with other
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param dt The collision dt (it may be larger than the simulation's dt).
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_ComputeUparTCorrection(koliop_Operator an_operator,
                                            double          dt,
                                            double*         f5d_s);

#define KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff  1
#define KOLIOP_EXTRACTION_IDENTIFIER_n_s           2
#define KOLIOP_EXTRACTION_IDENTIFIER_U_par_s       3
#define KOLIOP_EXTRACTION_IDENTIFIER_T_s           4
#define KOLIOP_EXTRACTION_IDENTIFIER_nu_ab         5
#define KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s 6

/// @brief Extract an internal buffer into the_output_buffer. The buffer in
/// question is specified using the_value_identifier and the
/// KOLIOP_EXTRACTION_IDENTIFIER_* preprocessor definition.
/// WARNING: Either call koliop_Fence after this call or chain with other
///
/// koliop_* operator's operations. You still may have to fence after the last
/// operation.
/// @param the_value_identifier A KOLIOP_EXTRACTION_IDENTIFIER_* preprocessor
/// definition.
/// @param the_output_buffer A correctly sized buffer.
koliop_Status koliop_ExtractRaw(koliop_Operator an_operator,
                                int32_t         the_value_identifier,
                                void*           an_output_buffer);

// TODO(Etienne M): Diags, not that making copies back into gysela could be
// unecessary. Could we avoid copies ? Should we do everything in koliop or not ?

// koliop_Status koliop_ExtractUrtheta(koliop_Operator an_operator,
//                                     double*         U_par_a_rtheta);

// koliop_Status koliop_ExtractQrtheta(koliop_Operator an_operator,
//                                     double*         q_par_);

// koliop_Status koliop_ExtractNuab(koliop_Operator an_operator,
//                                  double*         nu_ab);

/// @brief Computes the evolution of the distribution function due to
/// collisions. It computes the collision of all species against all other
/// species (if asked for through collision_interspecies and
/// the_species_extent). This "do it all method" is mostly used for the
/// Gyselalib++.
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
/// WARNING: The user has to koliop_Fence() to ensure the result of the operator
/// are visible for next operation.
///
/// @param dt The collision dt (it may be larger than the simulation's dt).
/// @param f5d_s An array of double with extents ([0, the_mu_extent), [0, the_vpar_extent), [0, the_r_extent), [0, the_theta_extent), [0, the_phi_extent), [0, the_species_extent) ).
/// @return KOLIOP_STATUS_SUCCESS on success.
koliop_Status koliop_Collision(koliop_Operator an_operator,
                               double          dt,
                               double*         f5d_s);

#if defined(__cplusplus)
}
#endif

#endif
