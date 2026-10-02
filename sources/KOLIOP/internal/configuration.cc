// SPDX-License-Identifier: MIT

#include "KOLIOP/internal/configuration.h"

#include <cstdio>

namespace koliop {
    namespace error {

        void PresentErrorExplanation(const char* a_category,
                                     const char* an_explanation) noexcept {
            std::fprintf(stderr,
                         "[KOLIOP][%s] %s\n",
                         a_category,
                         an_explanation);
        }

        void AssertionFailure(const char* message) noexcept {
            PresentErrorExplanation("ASSERTION_FAILURE", message);

            std::fflush(stdout);
            std::fflush(stderr);

            KOLIOP_DEBUG_BREAK();
        }

    } // namespace error
} // namespace koliop
