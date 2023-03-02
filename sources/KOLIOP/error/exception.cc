#include <cstdio>

namespace koliop {
    namespace error {

        void PresentExceptionExplanation(const char* a_category,
                                         const char* an_explanation) noexcept {
            std::printf("[INTERNAL_ERROR]:%s:%s",
                        a_category,
                        an_explanation);
        }

    } // namespace error
} // namespace koliop
