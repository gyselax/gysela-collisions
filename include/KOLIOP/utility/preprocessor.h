// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_PREPROCESSOR_H
#define KOLIOP_UTILITY_PREPROCESSOR_H

namespace koliop {
    namespace utility {

        ////////////////////////////////////////
        /// KOLIOP_UTILITY_STRINGIFY
        ///
        /// This macro will convert the text as argument to a string literal.
        ///
        /// Example usage:
        ///
        ///     KOLIOP_UTILITY_STRINGIFY(THIS IS A TEST);
        ///
        /// Expands to :
        ///
        ///     "THIS IS A TEST";
        ///
        ////////////////////////////////////////

#define _KOLIOP_UTILITY_STRINGIFY(an_expression) #an_expression
#define KOLIOP_UTILITY_STRINGIFY(an_expression)  _KOLIOP_UTILITY_STRINGIFY(an_expression)

    } // namespace utility
} // namespace koliop

#endif
