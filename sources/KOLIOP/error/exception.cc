// SPDX-License-Identifier: MIT

#include "KOLIOP/error/exception.h"

#include <cstdio>

namespace koliop {
    namespace error {

        void PresentErrorExplanation(const char* a_category,
                                     const char* an_explanation) noexcept {
            std::printf("[KOLIOP][%s] %s\n",
                        a_category,
                        an_explanation);
        }

        void AssertionFailure(const char* an_expression) noexcept {
            PresentErrorExplanation("ASSERTION_FAILURE",
                                    an_expression);
            std::fflush(stdout);
            std::fflush(stderr);

            KOLIOP_DEBUG_BREAK();
        }

    } // namespace error
} // namespace koliop
