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

koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
                            uint64_t         Npolmax,
                            uint64_t         index_max,
                            uint64_t         Nmu,
                            uint64_t         Nvpar,
                            uint64_t         Nspecies,
                            uint64_t         pSpecies_id,
                            uint64_t         mu_id,
                            uint64_t         istart,
                            uint64_t         iend,
                            uint64_t         jstart,
                            uint64_t         jend,
                            int64_t          collstart,
                            int64_t          collend,
                            double           R0,
                            uint32_t         collision_interspecies,
                            uint32_t         SOL,
                            uint32_t         LIM,
                            const double*    comb_mat,
                            uint64_t         comb_mat_extent_0,
                            uint64_t         comb_mat_extent_1,
                            const double*    hat_As,
                            uint64_t         hat_As_extent_0,
                            const double*    hat_Zs,
                            uint64_t         hat_Zs_extent_0);
koliop_Status koliop_Release(koliop_Operator an_operator);

koliop_Status koliop_PrepareNuStar(koliop_Operator an_operator);

#if defined(__cplusplus)
}
#endif

#endif
