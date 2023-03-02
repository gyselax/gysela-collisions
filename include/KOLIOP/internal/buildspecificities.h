#ifndef KOLIOP_INTERNAL_BUILDSPECIFICITIES_H
#define KOLIOP_INTERNAL_BUILDSPECIFICITIES_H

#define KOLIOP_VERSION_MAJOR 0
#define KOLIOP_VERSION_MINOR 0
#define KOLIOP_VERSION_PATCH 1
#define KOLIOP_VERSION_TWEAK 
#define KOLIOP_VERSION       "0.0.1"

////////////////////////////////////////
/// KOLIOP_ASSERT_ENABLED
///
/// If not defined, disable assertion, else, enable them.
///
/// Example usage:
///
///     #if defined(SPLB2_ASSERT_ENABLED)
///         KOLIOP_ASSERT(x > 10);
///     #else
///         something else;
///     #endif
///
////////////////////////////////////////

#define KOLIOP_ASSERT_ENABLED

#endif
