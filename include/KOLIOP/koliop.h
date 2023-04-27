// SPDX-License-Identifier: MIT

#ifndef KOLIOP_KOLIOP_H
#define KOLIOP_KOLIOP_H

// NOTE: Everything in this file should be C99.

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

/// @brief Has to be called before any other koliop function is called.
/// @param a_host_thread_count The host threads to use to drive the device or to
/// do the computation. Depending on how koliop was built.
/// @param a_device_index The device koliop will use to offload the work. The
/// value as no meaning if koliop was not built with device support enabled.
/// @param with_verbose_initialization KOLIOP_FALSE to disable or KOLIOP_TRUE to
/// present additional information.
/// @return KOLIOP_STATUS_SUCCESS on success TODO(Etienne M): more.
///
koliop_Status koliop_Initialize(uint32_t a_host_thread_count,
                                uint32_t a_device_index,
                                int32_t  with_verbose_initialization);

/// @brief Has to be called when koliop is not to be used anymore for the
/// lifetime of the program.
/// @return KOLIOP_STATUS_SUCCESS on success TODO(Etienne M): more.
///
koliop_Status koliop_Deinitialize(void);

/// @brief Represent a collision operator.
///
typedef struct _koliop_Operator* koliop_Operator;

/// The values given below are expected NOT to change during the lifetime
/// of the operator.
/// NOTE: Npolmax is expected to be 3 and index_max = 2 * (index_max - 1) = 4
/// NOTE: dvpar is used internally and is expected to be uniform.
/// TODO(Etienne M): remove vparg when we have the vpar sampler.
///
koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
                            uint64_t         Npolmax,
                            uint64_t         index_max,
                            uint64_t         pSpecies_id,
                            uint64_t         mu_id,
                            uint64_t         the_mu_extent,
                            uint64_t         the_vpar_extent,
                            uint64_t         the_r_extent,
                            uint64_t         the_theta_extent,
                            uint64_t         the_phi_extent,
                            uint64_t         the_species_extent,
                            double           R0,
                            uint32_t         collision_interspecies,
                            uint32_t         SOL,
                            uint32_t         LIM,
                            uint64_t         ir_SOL_separatrix,
                            double           dt,
                            double           dvpar,
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

/// @brief Computation of fluid moments term:
///   na =  int 2 * Pi * Bstar * dmu * dvpar * Fa
///   Vpara = (1/na) * int 2 * Pi * Bstar * dmu * dvpar * vpar * Fa
///   Ta = (1/(3*na)) * int 2 * Pi * Bstar * dmu * dvpar * Fa * &
///        (2 * mu *B + (vpar-Vpara)^2)
///
koliop_Status koliop_ComputeMoments(koliop_Operator an_operator,
                                    double*         f5d_s);

koliop_Status koliop_PrecomputeNu(koliop_Operator an_operator);

/// @brief Remove the Maxwellian to the distribution function Fa = Fa - FMa
/// Evolve temperature
/// Compute the new Maxwellian
/// Add it to the function distribution Fa = Fa + FMa
/// Evolution of the temperatures via the term C0:
///   Ta = Ta + deltat*sum_b nu_ab * (2*hat_Aa/(hat_Aa + hat_Ab)*(Tb -Ta) +
///     (2/3) * Vpara *(Vpara - Vparb))
///
koliop_Status koliop_ComputeMaxwellianEvolution(koliop_Operator an_operator,
                                                double*         f5d_s);

/// @brief Computation of q_para/(na*Ta)
///   q_par_on_nT = 0.5*int (2 * Pi * Bstar) *  dmu * dvpar * &
///       (vpar-Vpara) * ((vpar-Vpara)^2 + vperp^2) * (Fa - FMa)
///  Indeed  q_par_on_nT = 0 for a Maxwellian distribution function
///
koliop_Status koliop_ComputeQparOnNT(koliop_Operator an_operator,
                                     const double*   f5d_s);

/// @brief Evolution of f due to the Cpar
///   f = f + dt * sum_b (nu_sab(xb) * vpar / Ta) * F_M0a * &
///           (Rat * Vparb - Vpara + (q_para / (na * Ta)) * (1 - 2*xa*xa/5) -
///           (3/5) *  Rat * (q_parb / (nb * Tb)) / (1 + (vTa/vTb)^2 ))
///  with Rat = sqrt(hat_Aa/hat_Ab)
///   xa = v /(sqrt(2*Ta)) = sqrt(2*mu*B + vpar*vpar)/(sqrt(2*Ta))
///   xb = xa * vTa / vTb
///
koliop_Status koliop_ComputeCpar(koliop_Operator an_operator,
                                 double*         f5d_s);

/// @brief This routine compute one of the following operation on the
/// distribution function:
///  - choice 3 : Fa = Fa * FM0a
///  - choice 4 : Fa = Fa / FM0a
///
/// @param choice [3, 4]
///
koliop_Status koliop_ComputeMaxwellianUnshiftedLoc(koliop_Operator an_operator,
                                                   double*         f5d_s,
                                                   uint32_t        choice);

koliop_Status koliop_Apply(koliop_Operator an_operator,
                           double*         f5d_s);

koliop_Status koliop_ComputeDensityCorrection(koliop_Operator an_operator,
                                              double*         f5d_s);

koliop_Status koliop_ComputeUparTCorrection(koliop_Operator an_operator,
                                            double*         f5d_s);

// TODO(Etienne M): diags.

koliop_Status koliop_ExtractUrtheta(koliop_Operator an_operator,
                                    double*         U_par_a_rtheta);

koliop_Status koliop_ExtractQrtheta(koliop_Operator an_operator,
                                    double*         q_par_);

koliop_Status koliop_ExtractNuab(koliop_Operator an_operator,
                                 double*         nu_ab);

#if defined(__cplusplus)
}
#endif

#endif
