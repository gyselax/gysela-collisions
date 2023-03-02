#include "KOLIOP/internal/configuration.h"

#include <cstdio>

namespace koliop {

    void AssertionFailure(const char* an_expression) {
        std::fprintf(stderr, "%s\n", an_expression);
        std::fflush(stdout);
        std::fflush(stderr);

        KOLIOP_DEBUG_BREAK();
    }

} // namespace koliop
