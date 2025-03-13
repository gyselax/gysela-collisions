// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_FLATTEN_H
#define KOLIOP_UTILITY_FLATTEN_H

#include <Kokkos_Core.hpp>

#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
    #include "KOLIOP/utility/math.h"
#endif

namespace koliop {
    namespace utility {

        ////////////////////////////////////////////////////////////////////////
        // Flatten2D definition
        ////////////////////////////////////////////////////////////////////////

        /// Flattens two extents.
        ///
        template <typename Ordinal>
        class Flatten2D {
        public:
            using OrdinalType = Ordinal;

        public:
            KOKKOS_INLINE_FUNCTION constexpr Flatten2D(OrdinalType extent_0,
                                                       OrdinalType extent_1)
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                : extent_0_{CheapLog2(extent_0)}
                , extent_1_{CheapLog2(extent_1)} {
#else
                : extent_0_{extent_0}
                , extent_1_{extent_1} {
#endif
                // EMPTY
            }

            KOKKOS_INLINE_FUNCTION constexpr OrdinalType
            forward() const {
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                return (1 << extent_0_) << extent_1_;
#else
                return extent_0_ * extent_1_;
#endif
            }

            KOKKOS_INLINE_FUNCTION constexpr void
            backward(OrdinalType  flattened_index,
                     OrdinalType& extent_0_index,
                     OrdinalType& extent_1_index) const {
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                // extent_1_index = flattened_index >> extent_0_;
                // extent_0_index = flattened_index - extent_1_index * (1 << extent_0_);

                extent_0_index = flattened_index & ((1 << extent_0_) - 1);
                extent_1_index = flattened_index >> extent_0_;
#else
                // extent_1_index = flattened_index / extent_0_;
                // extent_0_index = flattened_index - extent_1_index * extent_0_;

                extent_0_index = flattened_index % extent_0_;
                extent_1_index = flattened_index / extent_0_;
#endif
            }

            template <typename Callable>
            KOKKOS_INLINE_FUNCTION constexpr void
            Call(OrdinalType flattened_index,
                 Callable&&  a_callable) const {
                OrdinalType extent_0_index;
                OrdinalType extent_1_index;
                backward(flattened_index,
                         extent_0_index,
                         extent_1_index);
                a_callable(extent_0_index,
                           extent_1_index);
            }

            // template <typename Callable>
            // KOKKOS_INLINE_FUNCTION constexpr auto
            // Wrap(Callable&& a_callable) const {
            //     return [&](auto flattened_index) {
            //         Call(flattened_index, a_callable);
            //     };
            // }

        protected:
            OrdinalType extent_0_;
            OrdinalType extent_1_;
        };

    } // namespace utility
} // namespace koliop

#endif
