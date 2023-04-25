// SPDX-License-Identifier: MIT

#ifndef KOLIOP_INTERNAL_CONFIGURATION_H
#define KOLIOP_INTERNAL_CONFIGURATION_H

#include "KOLIOP/internal/buildspecificities.h"
#include "KOLIOP/utility/preprocessor.h"


////////////////////////////////////////////////////////////////////////////////
// Platform macro definition
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////
/// KOLIOP_COMPILER_IS_ICX
/// KOLIOP_COMPILER_IS_ICC
/// KOLIOP_COMPILER_IS_CLANG
/// KOLIOP_COMPILER_IS_GCC
/// KOLIOP_COMPILER_IS_MSVC
///
/// Use these macros to test for compiler dependant features/source code...
/// <cpuid.h> (clang/gnu) vs <intrin.h> (msvc)
///
/// NOTE:
/// Clang (for instance) defines __GNUC__/_MSC_VER depending on the OS. That
/// means that some code path specialized using #ifdef, if badly ordered, will
/// prioritize an MSVC code path over a Clang code path. This can cause problem
/// with intrinsics or optimization.
///
/// Example:
///
///    #if defined(_MSC_VER)
///        static_assert(false, "");
///    #elif defined(__clang__)
///    #endif
///
/// Will not work on both clang and msvc under windows, the intent is that only
/// msvc should fail.
///
/// But:
///
///    #if defined(__clang__)
///    #elif defined(_MSC_VER)
///        static_assert(false, "");
///    #endif
/// Or:
///    #if !defined(__clang__) && defined(_MSC_VER)
///        static_assert(false, "");
///    #elif defined(__clang__)
///    #endif
///
/// Will work as expected.
///
/// This mimicry stuff helps porting code but, that is still a big PITA and
/// quite brittle. For this reason, we provide these PP definitions. The idea is
/// to check starting from the compilers *mimicking* other compilers the most to
/// the ones *mimicking* the less. By mimicking we mean defining say, __clang__
/// when in fact you are not the *real* Clang. For instance, Clang is used as a
/// base for the Intel Clang Compiler (ICL). So we need, before checking if the
/// compiler is Clang, to check that it not ICL. This goes on for all *__clang__
/// defining* compiler until we can finally say, we are using Clang and not a
/// derived product.
///
/// Example usage:
///     #if defined(KOLIOP_COMPILER_IS_CLANG)
///         // Clang (or mimicry/clang based) compiler code here.
///     #endif
///
/// See:
///     https://sourceforge.net/p/predef/wiki/Compilers/
///     https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html
///
/// Find out what is defined by default by your compiler:
///
/// Compiler                   C macros                         C++ macros
/// Clang/LLVM                 clang -dM -E -x c /dev/null      clang++ -dM -E -x c++ /dev/null
/// GNU GCC/G++                gcc   -dM -E -x c /dev/null      g++     -dM -E -x c++ /dev/null
/// Hewlett-Packard C/C++      cc    -dM -E -x c /dev/null      CC     -dM -E -x c++ /dev/null
/// IBM XL C/C++               xlc   -qshowmacros -E /dev/null  xlc++   -qshowmacros -E /dev/null
/// Intel ICC/ICPC             icc   -dM -E -x c /dev/null      icpc    -dM -E -x c++ /dev/null
/// Microsoft Visual Studio    https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=vs-2019
/// Oracle Solaris Studio      cc    -xdumpmacros -E /dev/null  CC      -xdumpmacros -E /dev/null
/// Portland Group PGCC/PGCPP  pgcc  -dM -E                     (none)
///
////////////////////////////////////////

#if defined(__INTEL_CLANG_COMPILER)
    // For icl
    #define KOLIOP_COMPILER_IS_ICX 1
#elif defined(__INTEL_COMPILER)
    // For icc and icl. The order matters, icl also defines __INTEL_COMPILER.
    #define KOLIOP_COMPILER_IS_ICC 1
#elif defined(__clang__)
    // For every Clang *compatible or mimicry* compiler. The order matters, icl also defines __clang__.
    #define KOLIOP_COMPILER_IS_CLANG 1
#elif defined(__GNUC__)
    // For every GCC *compatible or mimicry* compiler. The order matters, Clang also defines __GNUC__.
    #define KOLIOP_COMPILER_IS_GCC 1
#elif defined(_MSC_VER)
    // For every MSVC *compatible or mimicry* compiler.
    #define KOLIOP_COMPILER_IS_MSVC 1
#endif

#if !(defined(KOLIOP_COMPILER_IS_CLANG) || \
      defined(KOLIOP_COMPILER_IS_GCC) ||   \
      defined(KOLIOP_COMPILER_IS_MSVC))
// defined(KOLIOP_COMPILER_IS_ICX)
// defined(KOLIOP_COMPILER_IS_ICC)
static_assert(false, "This library doesn't support your compiler.");
#endif


////////////////////////////////////////////////////////////////////////////////
// Utility macro definition
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////
/// KOLIOP_DEBUG_BREAK
///
/// This macro will stop the program.
/// If it's beeing debugged, the program will be stoped.
/// If it's not beeing debugged, the program will crashdump.
///
/// Example usage:
///
///     KOLIOP_DEBUG_BREAK();
///
////////////////////////////////////////

#include <cstdlib>
// NOTE: We could also use __debugbreak or int3
#define KOLIOP_DEBUG_BREAK() std::exit(EXIT_FAILURE)


////////////////////////////////////////
/// KOLIOP_UNUSED
///
/// Prevent a compiler warning for an unused variable.
///
/// Example usage:
///
///     int a;
///     KOLIOP_UNUSED(a);
///     return;
///
////////////////////////////////////////

#define KOLIOP_UNUSED(an_expression) static_cast<void>(an_expression)


////////////////////////////////////////
/// KOLIOP_ASSERT
///
/// Use this macro to assert a condition. The assertions trigger only if
/// KOLIOP_ASSERT_ENABLED is defined.
///
/// Example usage:
///
///     KOLIOP_ASSERT(x > 10);
///
////////////////////////////////////////

namespace koliop {
    namespace error {
        void AssertionFailure(const char* the_message) noexcept;
    }
} // namespace koliop

#if defined(KOLIOP_ASSERT_ENABLED)
    #define KOLIOP_ASSERT(an_expression) KOLIOP_UNUSED((an_expression) || (koliop::error::AssertionFailure(#an_expression " in " __FILE__ ":" KOLIOP_UTILITY_STRINGIFY(__LINE__)), false))
#else
    #define KOLIOP_ASSERT(an_expression)
#endif


////////////////////////////////////////
/// KOLIOP_FORCE_INLINE
/// KOLIOP_FORCE_NOINLINE
///
/// Example usage:
///     KOLIOP_FORCE_INLINE inline int
///     foo(float* magic) { /* stuff */ return 1;}
///
////////////////////////////////////////

#if defined(KOLIOP_COMPILER_IS_GCC) || defined(KOLIOP_COMPILER_IS_CLANG)
    #define KOLIOP_FORCE_INLINE   __attribute__((always_inline))
    #define KOLIOP_FORCE_NOINLINE __attribute__((noinline))
#elif defined(KOLIOP_COMPILER_IS_MSVC)
    #define KOLIOP_FORCE_INLINE   __forceinline
    #define KOLIOP_FORCE_NOINLINE __declspec(noinline)
#endif


////////////////////////////////////////////////////////////////////////////////
/// Types
////////////////////////////////////////////////////////////////////////////////

namespace koliop {

    using Flo32 = float;
    using Flo64 = double;

    using ComputeFloatType = Flo64;
    using OrdinalType      = int64_t;
    using OffsetType       = uint64_t;

} // namespace koliop

#endif
