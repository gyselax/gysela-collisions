#ifndef KOLIOP_ERROR_STATUS_H
#define KOLIOP_ERROR_STATUS_H

// NOTE: Everything in this file should be C99.

#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
// koliop_Status definition
////////////////////////////////////////////////////////////////////////////////

typedef int32_t koliop_Status;

// NOTE: Errors go towards -infinity.

#define KOLIOP_STATUS_SUCCESS                  0
#define KOLIOP_STATUS_UNKOWN_FAILURE           -1
#define KOLIOP_STATUS_INITIALIZATION_FAILURE   -2
#define KOLIOP_STATUS_DEINITIALIZATION_FAILURE -3
#define KOLIOP_STATUS_INVALID_VALUE            -30

// #define KOLIOP_STATUS_DEVICE_NOT_FOUND         -1000
// #define KOLIOP_STATUS_ALLOCATION_FAILURE       -1000
// #define KOLIOP_STATUS_OUT_OF_XXXXX             -1000
// #define KOLIOP_STATUS_INVALID_HOST_POINTER     -1000
// #define KOLIOP_STATUS_INVALID_DEVICE_POINTER   -1000
// #define KOLIOP_STATUS_INVALID_WORK_DIMENSION   -1000
// #define KOLIOP_STATUS_INVALID_OPERATION        -1000
// #define KOLIOP_STATUS_INVALID_BUFFER_SIZE      -1000
// #define KOLIOP_STATUS_INVALID_GLOBAL_WORK_SIZE -1000
// #define KOLIOP_STATUS_INVALID_PROPERTY         -1000

#endif
