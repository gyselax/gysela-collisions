#ifndef KOLIOP_ERROR_EXCEPTION_H
#define KOLIOP_ERROR_EXCEPTION_H

#include <stdexcept>

#include "KOLIOP/error/status.h"
#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace error {

        void PresentErrorExplanation(const char* a_category,
                                     const char* an_explanation) noexcept;

        template <typename Callable>
        ::koliop_Status
        FromException(Callable&& a_callable) noexcept {
            try {
                return a_callable();
                // } catch(const XXXXXXX&) {
                //     return KOLIOP_STATUS_XXXXXXX;
            } catch(const std::runtime_error& an_exception) {
                PresentErrorExplanation("std::runtime_error",
                                        an_exception.what());
                return KOLIOP_STATUS_FAILURE;
            } catch(const std::exception& an_exception) {
                PresentErrorExplanation("std::exception",
                                        an_exception.what());
                return KOLIOP_STATUS_FAILURE;
            } catch(...) {
                KOLIOP_ASSERT(false);
                return KOLIOP_STATUS_FAILURE;
            }

            KOLIOP_ASSERT(false);
            return KOLIOP_STATUS_FAILURE;
        }
    } // namespace error
} // namespace koliop

#endif
