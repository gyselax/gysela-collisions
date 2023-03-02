#include "MYLIB/internal/configuration.h"

#include <cstdio>
#include <cstdlib>

namespace mylib {

    void AssertionFailure(const char* an_expression) {
        std::fprintf(stderr, "%s\n", an_expression);
        std::fflush(stdout);
        std::fflush(stderr);

        std::exit(EXIT_FAILURE); // Or a debug break.
    }

} // namespace mylib
