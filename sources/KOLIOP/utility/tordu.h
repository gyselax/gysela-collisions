// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_TORDU_H
#define KOLIOP_UTILITY_TORDU_H

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace koliop {
    namespace utility {
        namespace detail {

            /// @brief Koliop (like Gysela fortran), expects the space extents
            /// like [(vpar/mu), r, theta, phi] with fast contiguous access on
            /// the left most indices.
            /// The C++ code (geezelalib++) assumes [theta, r, phi].
            /// The collision operator is only in vpar, mu, the r/theta/phi are
            /// parameters but some loops read data that is dependent on say, r.
            ///
            /// In Fortran:
            ///     tor1, tor2, tor3 = r, theta, phi
            ///     f(mu, vpar, r, theta, phi)
            /// In C++:
            ///     tor1, tor2, tor3 = theta, r, phi
            ///     f(mu, vpar, theta, r, phi)
            ///
            /// We iterate like so:
            ///     for(tor1, tor2, tor3) {
            ///         fd5d_s(mu, vpar, tor1, tor2, tor3) *= AD(r)
            ///         fd5d_s(mu, vpar, tor1, tor2, tor3) *= B_norm(tor1, tor2)
            ///     }
            /// How to get r from (tor1,tor2,tor3) ?
            /// To make this work, we have to know the mapping from torN to
            /// (r,theta,phi). We need to carefully specify
            ///
            template <int kAt,
                      typename Value0,
                      typename Value1,
                      typename Value2>
            KOKKOS_INLINE_FUNCTION constexpr auto
            GetTorValue(Value0&& tor1_value,
                        Value1&& tor2_value,
                        Value2&& tor3_value) {
#if defined(KOLIOP_BUILD_FORTRAN_INTERFACE)
                // Gysela fortran
                constexpr int kMapping[]{0, 1, 2};
#else
                // Gyselalib++
                constexpr int kMapping[]{1, 0, 2};
#endif

                if constexpr(kMapping[kAt] == 0) {
                    return tor1_value;
                } else if constexpr(kMapping[kAt] == 1) {
                    return tor2_value;
                } else if constexpr(kMapping[kAt] == 2) {
                    return tor3_value;
                } else {
                    static_assert(false);
                }
            }

        } // namespace detail

        /// Deduces the right value for R given tree, "erased" value.
        ///
        template <typename Value0,
                  typename Value1,
                  typename Value2>
        KOKKOS_INLINE_FUNCTION constexpr auto
        GetRValue(Value0&& tor1_value,
                  Value1&& tor2_value,
                  Value2&& tor3_value) {
            return detail::GetTorValue<0>(tor1_value, tor2_value, tor3_value);
        }

        /// Deduces the right value for Theta given tree, "erased" value.
        ///
        template <typename Value0,
                  typename Value1,
                  typename Value2>
        KOKKOS_INLINE_FUNCTION constexpr auto
        GetThetaValue(Value0&& tor1_value,
                      Value1&& tor2_value,
                      Value2&& tor3_value) {
            return detail::GetTorValue<1>(tor1_value, tor2_value, tor3_value);
        }

        /// Deduces the right value for Phi given tree, "erased" value.
        ///
        template <typename Value0,
                  typename Value1,
                  typename Value2>
        KOKKOS_INLINE_FUNCTION constexpr auto
        GetPhiValue(Value0&& tor1_value,
                    Value1&& tor2_value,
                    Value2&& tor3_value) {
            return detail::GetTorValue<2>(tor1_value, tor2_value, tor3_value);
        }

    } // namespace utility
} // namespace koliop

#endif
