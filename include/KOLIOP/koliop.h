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

/// The values given below are expected NOT to change during the lifetime
/// of the operator.
/// NOTE: Npolmax is expected to be 3 and index_max = 2 * (index_max - 1) = 4
///
koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
                            size_t           Npolmax,
                            size_t           index_max,
                            int64_t          pSpecies_id,
                            int64_t          mu_id,
                            size_t           the_mu_extent,
                            size_t           the_vpar_extent,
                            size_t           the_r_extent,
                            size_t           the_theta_extent,
                            size_t           the_phi_extent,
                            size_t           the_species_extent,
                            double           R0,
                            int8_t           collision_interspecies,
                            int8_t           SOL,
                            int8_t           LIM,
                            int64_t          ir_SOL_separatrix,
                            const double*    comb_mat,
                            const double*    hat_As,
                            const double*    hat_Zs,
                            const double*    mug,
                            const double*    vparg,
                            const double*    coeff_intdmu,
                            const double*    coeff_intdvpar,
                            const double*    rg,
                            const double*    nustar0_r,
                            const double*    q_r,
                            const double*    mask_buffer_r,
                            const double*    mask_LIM,
                            const double*    B_norm,
                            const double*    Bstar_s);

koliop_Status koliop_Release(koliop_Operator an_operator);

koliop_Status koliop_Fence(koliop_Operator an_operator);

/// Computation of fluid moments term:
///   na =  int 2 * Pi * Bstar * dmu * dvpar * Fa
///   Vpara = (1/na) * int 2 * Pi * Bstar * dmu * dvpar * vpar * Fa
///   Ta = (1/(3*na)) * int 2 * Pi * Bstar * dmu * dvpar * Fa * &
///        (2 * mu *B + (vpar-Vpara)^2)
///
koliop_Status koliop_ComputeMoments(koliop_Operator an_operator,
                                    double*         f5d_s);

koliop_Status koliop_PrecomputeNu(koliop_Operator an_operator);

/// Remove the Maxwellian to the distribution function Fa = Fa - FMa
/// Evolve temperature
/// Compute the new Maxwellian
/// Add it to the function distribution Fa = Fa + FMa
/// Evolution of the temperatures via the term C0:
///   Ta = Ta + deltat*sum_b nu_ab * (2*hat_Aa/(hat_Aa + hat_Ab)*(Tb -Ta) +
///     (2/3) * Vpara *(Vpara - Vparb))
///
koliop_Status koliop_ComputeMaxwellianEvolution(koliop_Operator an_operator,
                                                double          dt,
                                                double*         f5d_s);

/// Computation of q_para/(na*Ta)
///   q_par_on_nT = 0.5*int (2 * Pi * Bstar) *  dmu * dvpar * &
///       (vpar-Vpara) * ((vpar-Vpara)^2 + vperp^2) * (Fa - FMa)
///  Indeed  q_par_on_nT = 0 for a Maxwellian distribution function
///
koliop_Status koliop_ComputeQparOnNT(koliop_Operator an_operator,
                                     const double*   f5d_s);

/// Evolution of f due to the Cpar
///   f = f + dt * sum_b (nu_sab(xb) * vpar / Ta) * F_M0a * &
///           (Rat * Vparb - Vpara + (q_para / (na * Ta)) * (1 - 2*xa*xa/5) -
///           (3/5) *  Rat * (q_parb / (nb * Tb)) / (1 + (vTa/vTb)^2 ))
///  with Rat = sqrt(hat_Aa/hat_Ab)
///   xa = v /(sqrt(2*Ta)) = sqrt(2*mu*B + vpar*vpar)/(sqrt(2*Ta))
///   xb = xa * vTa / vTb
///
koliop_Status koliop_ComputeCpar(koliop_Operator an_operator,
                                 double          dt,
                                 double*         f5d_s);

/// This routine compute one of the following operation on the
/// distribution function:
///  - choice 3 : Fa = Fa * FM0a
///  - choice 4 : Fa = Fa / FM0a
///
/// @param choice [3, 4]
///
koliop_Status koliop_ComputeMaxwellianUnshiftedLoc(koliop_Operator an_operator,
                                                   double*         f5d_s,
                                                   int32_t         choice);

koliop_Status koliop_Apply(koliop_Operator an_operator,
                           double          dt,
                           double*         f5d_s);

koliop_Status koliop_ComputeDensityCorrection(koliop_Operator an_operator,
                                              double*         f5d_s);

koliop_Status koliop_ComputeUparTCorrection(koliop_Operator an_operator,
                                            double          dt,
                                            double*         f5d_s);

#define KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff 1
#define KOLIOP_EXTRACTION_IDENTIFIER_n_s 2
#define KOLIOP_EXTRACTION_IDENTIFIER_U_par_s 3
#define KOLIOP_EXTRACTION_IDENTIFIER_T_s 4
#define KOLIOP_EXTRACTION_IDENTIFIER_nu_ab 5
#define KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s 6

/// @brief Extract an internal buffer into the_output_buffer. The buffer in
/// question is specified using the_value_identifier and the
/// KOLIOP_EXTRACTION_IDENTIFIER_* preprocessor definition.
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

#if defined(__cplusplus)
}
#endif

#endif
