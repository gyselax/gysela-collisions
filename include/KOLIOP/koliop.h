#ifndef KOLIOP_KOLIOP_H
#define KOLIOP_KOLIOP_H

// NOTE: Everything in this file should be C99.

#include <stdint.h>

#include "KOLIOP/error/status.h"

////////////////////////////////////////////////////////////////////////////////
// koliop method declaration
////////////////////////////////////////////////////////////////////////////////

#if defined(__cplusplus)
extern "C" {
#endif

/// @brief Has to be called before any other koliop function is called.
/// @param a_host_thread_count The host threads to use to drive the device or to
/// do the computation. Depending on how koliop was built.
/// @param a_device_index The device koliop will use to offload the work (if
/// build with device support).
/// @param with_verbose_initialization 0 to disable, else, present additional
/// information. TODO(Etienne M): document
/// @return KOLIOP_STATUS_SUCCESS on success
koliop_Status koliop_Initialize(uint32_t a_host_thread_count,
                                uint32_t a_device_index,
                                int32_t  with_verbose_initialization);

/// @brief Has to be called when koliop is not to be used anymore for the
/// lifetime of the program.
/// @return KOLIOP_STATUS_SUCCESS on success
koliop_Status koliop_Deinitialize(void);

#if defined(__cplusplus)
}
#endif

#endif
