#ifndef KOLIOP_ERROR_EXCEPTION_H
#define KOLIOP_ERROR_EXCEPTION_H

#include "KOLIOP/error/status.h"
#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace error {

        template <typename Callable>
        ::koliop_Status
        FromException(Callable&& a_callable) noexcept {
            try {
                return a_callable();
                // } catch(const XXXXXXX&) {
                //     return KOLIOP_STATUS_XXXXXXX;
            } catch(const std::exception&) {
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
