#ifndef MYLIB_INTERNAL_CONFIGURATION_H
#define MYLIB_INTERNAL_CONFIGURATION_H

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "MYLIB/internal/buildspecificities.h"
#include "MYLIB/utility/preprocessor.h"

////////////////////////////////////////////////////////////////////////////////
// Platform macro definition
////////////////////////////////////////////////////////////////////////////////

#if !(defined(__clang__) || defined(__GNUC__) || defined(_MSC_VER) && (_MSC_VER >= 1300))
static_assert(false, "This library doesn't support your compiler.");
#endif


//////////////////////////////////////
/// DOXYGEN_IS_DOCUMENTING
///
/// Use this macro to test if you should define an empty macro just for the doc generation
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define DOXYGEN_IS_DOCUMENTING 1
#endif


//////////////////////////////////////
/// MYLIB_ARCH_IS_X64
/// MYLIB_ARCH_IS_ARM64
///
/// Use these to test for the current architecture we are compiling for
/// https://sourceforge.net/p/predef/wiki/Architectures/
/// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-160
///
/// Example usage:
///
///     #if defined(MYLIB_ARCH_IS_X64)
///         // MYLIB_ARCH_IS_X64 code here
///     #endif
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_ARCH_IS_X64
    #define MYLIB_ARCH_IS_ARM64
#else
    #if defined(__amd64__) || defined(_M_X64)
        #define MYLIB_ARCH_IS_X64 1
    #elif defined(__aarch64__) || defined(_M_ARM64)
        #define MYLIB_ARCH_IS_ARM64 1
    #else
static_assert(false, "This library doesn't support the architecture you are compiling for.");
    #endif
#endif


//////////////////////////////////////
/// MYLIB_OS_IS_LINUX
/// MYLIB_OS_IS_WINDOWS
///
/// Use these macros to test for platform dependant code (try to avoid that).
///
/// Example usage:
///
///     #if defined(MYLIB_OS_IS_LINUX)
///         // Linux code here
///     #endif
///
/// We could use MYLIB_PLATFORM_IS.
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_OS_IS_LINUX
    #define MYLIB_OS_IS_WINDOWS
#else
    #if defined(_WIN32)
        #define MYLIB_OS_IS_WINDOWS 1
    #elif defined(__linux__)
        #define MYLIB_OS_IS_LINUX 1
    #else
static_assert(false, "This library doest not support your OS.");
    #endif
#endif


////////////////////////////////////////////////////////////////////////////////
// Utility macro definition
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////
/// MYLIB_DEBUG_BREAK
///
/// This macro will stop the program.
/// If it's beeing debugged, the program will be stoped.
/// If it's not beeing debugged, the program will crashdump.
///
/// Example usage:
///
///     MYLIB_DEBUG_BREAK();
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_DEBUG_BREAK()
#else
    #if defined(__GNUC__) || defined(__clang__)
        #define MYLIB_DEBUG_BREAK() __asm__("int3")
    #elif defined(_MSC_VER)
        #define MYLIB_DEBUG_BREAK() __debugbreak()
    #else
        #include <cstdlib>
        #define MYLIB_DEBUG_BREAK() std::exit(EXIT_FAILURE)
    #endif
#endif


//////////////////////////////////////
/// MYLIB_ASSERT_ENABLED
///
/// If not defined, assertions are disabled, else, they are enabled. Do not
/// define that manually, use the CMake option instead.
///
/// Example usage:
///
///     #if defined(MYLIB_ASSERT_ENABLED)
///         MYLIB_ASSERT(x > 10);
///     #else
///         something else;
///     #endif
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_ASSERT_ENABLED
#endif


//////////////////////////////////////
/// MYLIB_UNUSED
///
/// Prevent a compiler warning for an unused variable.
///
/// Example usage:
///
///     int a;
///     MYLIB_UNUSED(a);
///     return;
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_UNUSED(an_expression)
#else
    #define MYLIB_UNUSED(an_expression) static_cast<void>(an_expression)
#endif


//////////////////////////////////////
/// MYLIB_FAIL_MESSAGE
///
/// Use this macro to report a failure.
///
/// Example usage:
///
///     if(x.empty()) { MYLIB_FAIL_MESSAGE("Empty x"); }
///
//////////////////////////////////////

namespace mylib {
    void AssertionFailure(const char* an_expression);
} // namespace mylib

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_FAIL_MESSAGE(an_expression)
#else
    #define MYLIB_FAIL_MESSAGE(an_expression) mylib::AssertionFailure(an_expression)
#endif


//////////////////////////////////////
/// MYLIB_ASSERT
///
/// Use this macro to assert a condition.
///
/// Example usage:
///
///     MYLIB_ASSERT(x > 10);
///     // If MYLIB_ASSERT_ENABLED is defined, this is equivalent to
///     // MYLIB_ASSERT, else, it execute an_expression but never trigger
///     // assertion failures.
///
//////////////////////////////////////

#if defined(DOXYGEN_IS_DOCUMENTING)
    #define MYLIB_ASSERT(an_expression)
#else
    #if defined(MYLIB_ASSERT_ENABLED)
        #define MYLIB_ASSERT(an_expression) MYLIB_UNUSED((an_expression) || (MYLIB_FAIL_MESSAGE(#an_expression " in " __FILE__ ":" MYLIB_UTILITY_STRINGIFY(__LINE__)), false))
    #else
        #define MYLIB_ASSERT(an_expression)
    #endif
#endif


////////////////////////////////////////////////////////////////////////////////
/// Types
////////////////////////////////////////////////////////////////////////////////

namespace mylib {

    using Uint64 = ::uint64_t;
    using Uint32 = ::uint32_t;
    using Uint16 = ::uint16_t;
    using Uint8  = ::uint8_t;

    using Int64 = ::int64_t;
    using Int32 = ::int32_t;
    using Int16 = ::int16_t;
    using Int8  = ::int8_t;

    using Flo32 = float;
    using Flo64 = double;

    using PointerHolderType = ::uintptr_t;
    using DifferenceType    = ::ptrdiff_t;
    using SizeType          = ::size_t;
    using SignedSizeType    = DifferenceType;

} // namespace mylib

#endif
