#ifndef KOLIOP_ERROR_STATUS_H
#define KOLIOP_ERROR_STATUS_H

// NOTE: Everything in this file should be C99.

#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
// koliop_Status definition
////////////////////////////////////////////////////////////////////////////////

typedef int32_t koliop_Status;

// NOTE: Errors go towards -infinity.

#define KOLIOP_STATUS_SUCCESS 0
#define KOLIOP_STATUS_FAILURE -1

#endif
