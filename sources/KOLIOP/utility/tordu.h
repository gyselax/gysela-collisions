// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_TORDU_H
#define KOLIOP_UTILITY_TORDU_H

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace koliop {
    namespace utility {
        namespace detail {

            template <int kTorIndex>
            KOKKOS_INLINE_FUNCTION constexpr auto
            GetMapping() {
                // Gyselalib++
                constexpr int kMapping[]{1, 0, 2};

                return kMapping[kTorIndex];
            }

            /// @brief kDimensionIndex is:
            ///     0 for R, 1 fo r Theta, 2 for Phi.
            ///
            template <int kDimensionIndex,
                      int kTorIdentifier,
                      typename Value0,
                      typename Value1>
            KOKKOS_INLINE_FUNCTION constexpr auto
            SetValue(Value0&& current_tor_value,
                     Value1&& new_tor_value) {
                if(detail::GetMapping<kDimensionIndex>() == kTorIdentifier) {
                    return new_tor_value;
                } else {
                    return current_tor_value;
                }
            }

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
            template <int kTorIndex,
                      typename Value0,
                      typename Value1,
                      typename Value2>
            KOKKOS_INLINE_FUNCTION constexpr auto
            GetTorValue(Value0&& tor1_value,
                        Value1&& tor2_value,
                        Value2&& tor3_value) {
                if constexpr(GetMapping<kTorIndex>() == 0) {
                    return tor1_value;
                } else if constexpr(GetMapping<kTorIndex>() == 1) {
                    return tor2_value;
                } else if constexpr(GetMapping<kTorIndex>() == 2) {
                    return tor3_value;
                } else {
                    // NOTE: the program is ill formed if I use
                    // static_assert(false) here regardless of the input
                    // template parameter. In truth this feels like a C++ defect
                    // because the intent of if constexpr in our case is to
                    // avoid SFINAE.. Anyway, some compiler accept this
                    // construct (gcc > 11 or clang) some older do not. I
                    // workaround the issue I generate a tautology based on the
                    // template argument.
                    static_assert(kTorIndex || !kTorIndex);
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

        template <int kTorIdentifier,
                  typename Value0,
                  typename Value1>
        KOKKOS_INLINE_FUNCTION constexpr auto
        SetRValue(Value0&& current_tor_value,
                  Value1&& new_tor_value) {
            return detail::SetValue<0, kTorIdentifier>(current_tor_value, new_tor_value);
        }

        template <int kTorIdentifier,
                  typename Value0,
                  typename Value1>
        KOKKOS_INLINE_FUNCTION constexpr auto
        SetThetaValue(Value0&& current_tor_value,
                      Value1&& new_tor_value) {
            return detail::SetValue<1, kTorIdentifier>(current_tor_value, new_tor_value);
        }

        template <int kTorIdentifier,
                  typename Value0,
                  typename Value1>
        KOKKOS_INLINE_FUNCTION constexpr auto
        SetPhiValue(Value0&& current_tor_value,
                    Value1&& new_tor_value) {
            return detail::SetValue<2, kTorIdentifier>(current_tor_value, new_tor_value);
        }

    } // namespace utility
} // namespace koliop

#endif
