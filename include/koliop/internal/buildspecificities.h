#ifndef MYLIB_INTERNAL_BUILDSPECIFICITIES_H
#define MYLIB_INTERNAL_BUILDSPECIFICITIES_H

#define MYLIB_VERSION_MAJOR 0
#define MYLIB_VERSION_MINOR 0
#define MYLIB_VERSION_PATCH 1
#define MYLIB_VERSION_TWEAK 
#define MYLIB_VERSION       "0.0.1"

// // Some values are exposed here:
// // https://gitlab.kitware.com/cmake/cmake/-/issues/21489
// #define MYLIB_PLATFORM_IS "Linux"

// // No always accurate on for windows build. (A microsoft fkup).
// #define MYLIB_SYSTEM_PROCESSOR "x86_64"

// #define MYLIB_SIZEOF_DATA_POINTER 8

#define MYLIB_ASSERT_ENABLED
/* #undef MYLIB_OPENMP_ENABLED */
/* #undef MYLIB_MPI_ENABLED */
/* #undef MYLIB_HIP_ENABLED */

#endif
