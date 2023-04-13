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

koliop_Status koliop_ComputeMoments(koliop_Operator an_operator,
                                    double*         f5d_s);

koliop_Status koliop_PrecomputeNu(koliop_Operator an_operator);

koliop_Status koliop_ComputeMaxwellianEvolution(koliop_Operator an_operator,
                                                double*         f5d_s);

koliop_Status koliop_ComputeQparOnNT(koliop_Operator an_operator,
                                     const double*   f5d_s);

koliop_Status koliop_ComputeCpar(koliop_Operator an_operator,
                                 double*         f5d_s);

// TODO(Etienne M): Document choice
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
