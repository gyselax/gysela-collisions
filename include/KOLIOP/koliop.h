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

/// @brief The values given below are expected NOT to change during the lifetime
/// of the operator. Notably, comb_mat, hat_As, hat_Zs, mug, vparg etc.
/// @param an_operator_handle
/// @param Npolmax
/// @param index_max
/// @param Nmu
/// @param Nvpar
/// @param Nspecies
/// @param pSpecies_id
/// @param mu_id
/// @param istart
/// @param iend
/// @param jstart
/// @param jend
/// @param collstart
/// @param collend
/// @param R0
/// @param collision_interspecies
/// @param SOL
/// @param LIM
/// @param comb_mat
/// @param comb_mat_extent_0
/// @param comb_mat_extent_1
/// @param hat_As
/// @param hat_As_extent_0
/// @param hat_Zs
/// @param hat_Zs_extent_0
/// @param mug
/// @param mug_extent_0
/// @param vparg
/// @param vparg_extent_0
/// @param coeff_intdmu
/// @param coeff_intdmu_extent_0
/// @param coeff_intdvpar
/// @param coeff_intdvpar_extent_0
/// @return
koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
                            uint64_t         Npolmax,
                            uint64_t         index_max,
                            uint64_t         Nmu,
                            uint64_t         Nvpar,
                            uint64_t         Nspecies,
                            uint64_t         pSpecies_id,
                            uint64_t         mu_id,
                            uint64_t         the_r_extent,
                            uint64_t         the_theta_extent,
                            uint64_t         the_phi_extent,
                            double           R0,
                            uint32_t         collision_interspecies,
                            uint32_t         SOL,
                            uint32_t         LIM,
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
                            const double*    B_norm,
                            const double*    Bstar_s);
koliop_Status koliop_Release(koliop_Operator an_operator);

koliop_Status koliop_ComputeMoments(koliop_Operator an_operator,
                                    double*         f5d_s);

// TODO(Etienne M): nu_ab becomes internal

koliop_Status koliop_ComputeFrequencyNuab(koliop_Operator an_operator,
                                          double          nustar_coeff,
                                          uint64_t        speciesA,
                                          uint64_t        speciesB,
                                          double          Ta,
                                          double          nb,
                                          double          Tb,
                                          double*         nu_ab);

koliop_Status koliop_ComputeEvolutionMaxwellian(koliop_Operator an_operator,
                                                double*         f5d_s);

koliop_Status koliop_ComputeQparOnNT(koliop_Operator an_operator,
                                     const double* f5d_s);

koliop_Status koliop_ComputeCpar(koliop_Operator an_operator,
                                 double*         f5d_s);

koliop_Status koliop_ComputeOperationMaxwellianUnshiftedLoc(koliop_Operator an_operator,
                                                            const double*   B_norm, // TODO(Etienne M): const external
                                                            double*         f5d_s,
                                                            const double*   n_s,
                                                            const double*   T_s,
                                                            uint32_t        choice); // external

// TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
// we have to do something similar.

// TODO(Etienne M): we can drop the ir_start/ir_end and just pass the range as
// the arrays are not subarrays.

koliop_Status koliop_ComputeAlphaProjection(koliop_Operator an_operator,
                                            uint64_t        ir_start,
                                            uint64_t        ir_end,
                                            uint64_t        itheta_start,
                                            uint64_t        itheta_end,
                                            uint64_t        iphi_start,
                                            uint64_t        iphi_end,
                                            // mask_LIM, // const external
                                            const uint8_t* mask_coll, // TODO(Etienne M): internal but needs mask_LIM
                                            const double*  B_norm,    // TODO(Etienne M): const external
                                            const double*  f5d_s,
                                            const double*  T_s, // TODO(Etienne M): const external
                                            double*        RHS_vec);   // TODO(Etienne M): internal

koliop_Status koliop_ComputeBlockTo1d(koliop_Operator an_operator,
                                      uint64_t        ir_start,
                                      uint64_t        ir_end,
                                      uint64_t        itheta_start,
                                      uint64_t        itheta_end,
                                      uint64_t        iphi_start,
                                      uint64_t        iphi_end,
                                      const uint8_t*  mask_coll,
                                      const double*   nustar_coeff,
                                      const double*   n_s,
                                      const double*   U_par_s,
                                      const double*   T_s,
                                      const double*   q_par_on_nT_s,
                                      double*         n1d_s,         // TODO(Etienne M): internal
                                      double*         T1d_s,         // TODO(Etienne M): internal
                                      double*         U_par1d,       // TODO(Etienne M): internal
                                      double*         q_par_on_nT1d, // TODO(Etienne M): internal
                                      double*         nu1d);                 // TODO(Etienne M): internal

koliop_Status koliop_ComputeCvCd(koliop_Operator an_operator,
                                 uint64_t        Nspace,
                                 const double*   n1d_s,
                                 const double*   T1d_s,
                                 const double*   U_par1d,
                                 const double*   q_par_on_nT1d,
                                 const double*   nu1d,
                                 double*         RHS_vec,
                                 double*         N0,      // TODO(Etienne M): internal
                                 double*         N1,      // TODO(Etienne M): internal
                                 double*         N2,      // TODO(Etienne M): internal
                                 double*         D_mat,   // TODO(Etienne M): internal
                                 double*         DL_mat,  // TODO(Etienne M): internal
                                 double*         DU1_mat, // TODO(Etienne M): internal
                                 double*         DU2_mat, // TODO(Etienne M): internal
                                 double          dt);

koliop_Status koliop_ComputeReconstruction(koliop_Operator an_operator,
                                           uint64_t        ir_start,
                                           uint64_t        ir_end,
                                           uint64_t        itheta_start,
                                           uint64_t        itheta_end,
                                           uint64_t        iphi_start,
                                           uint64_t        iphi_end,
                                           const uint8_t*  mask_coll,
                                           const double*   B_norm,
                                           double*         f5d_s,
                                           const double*   T_s,
                                           const double*   RHS_vec);

koliop_Status koliop_ComputeCorrectionDensity(koliop_Operator an_operator,
                                              const double*   Bstar_s,
                                              double*         f5d_s,
                                              const double*   n_s);

koliop_Status koliop_ComputeCorrectionUparT(koliop_Operator an_operator,
                                            const double*   nustar_coeff,
                                            const double*   B_norm,
                                            const double*   Bstar_s,
                                            double*         f5d_s,
                                            const double*   n_s,
                                            double*         U_par_s,
                                            const double*   T_s,
                                            const double*   q_par_on_nT_s,
                                            double          dt);

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
