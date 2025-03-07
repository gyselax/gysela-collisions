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
        // Flatten definition
        ////////////////////////////////////////////////////////////////////////

        /// Flattens two extents.
        ///
        template <typename Ordinal>
        class Flatten2 {
        public:
            using OrdinalType = Ordinal;

        public:
            KOKKOS_INLINE_FUNCTION constexpr Flatten2(OrdinalType extent_1,
                                                      OrdinalType extent_2)
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                : extent_1_{CheapLog2(extent_1)}
                , extent_2_{CheapLog2(extent_2)} {
#else
                : extent_1_{extent_1}
                , extent_2_{extent_2} {
#endif
                // EMPTY
            }

            KOKKOS_INLINE_FUNCTION constexpr OrdinalType
            forward() const {
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                return (1 << extent_1_) << extent_2_;
#else
                return extent_1_ * extent_2_;
#endif
            }

            KOKKOS_INLINE_FUNCTION constexpr void
            backward(OrdinalType  flattened_index,
                     OrdinalType& extent_1_index,
                     OrdinalType& extent_2_index) const {
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                extent_1_index = (flattened_index) & ((1 << extent_1_) - 1);
                extent_2_index = (flattened_index >> (extent_1_)) /* % extent_2_ */;
#else
                extent_1_index = (flattened_index) % extent_1_;
                extent_2_index = (flattened_index / (extent_1_)) /* % extent_2_ */;
#endif
            }

            template <typename Callable>
            KOKKOS_INLINE_FUNCTION constexpr void
            Call(OrdinalType flattened_index,
                 Callable&&  a_callable) const {
                OrdinalType extent_1_index;
                OrdinalType extent_2_index;
                backward(flattened_index,
                         extent_1_index,
                         extent_2_index);
                a_callable(extent_1_index,
                           extent_2_index);
            }

        protected:
            OrdinalType extent_1_;
            OrdinalType extent_2_;
        };

    } // namespace utility
} // namespace koliop

#endif
