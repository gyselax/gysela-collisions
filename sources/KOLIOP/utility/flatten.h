// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_FLATTEN_H
#define KOLIOP_UTILITY_FLATTEN_H

#include <Kokkos_Core.hpp>

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
                : extent_1_{extent_1}
                , extent_2_{extent_2} {
                // EMPTY
            }

            KOKKOS_INLINE_FUNCTION constexpr OrdinalType
            forward() const {
                return extent_1_ * extent_2_;
            }

            KOKKOS_INLINE_FUNCTION constexpr void
            backward(OrdinalType  flattened_index,
                     OrdinalType& extent_1_index,
                     OrdinalType& extent_2_index) const {
                // extent_1_index = (flattened_index) & (extent_1_ - 1);
                extent_1_index = (flattened_index) % extent_1_;
                extent_2_index = (flattened_index / (extent_1_)) /* % extent_2_ */;
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

        /// Flattens three extents.
        ///
        template <typename Ordinal>
        class Flatten3 {
        public:
            using OrdinalType = Ordinal;

        public:
            KOKKOS_INLINE_FUNCTION constexpr Flatten3(OrdinalType extent_1,
                                                      OrdinalType extent_2,
                                                      OrdinalType extent_3)
                : extent_1_{extent_1}
                , extent_2_{extent_2}
                , extent_3_{extent_3} {
                // EMPTY
            }

            KOKKOS_INLINE_FUNCTION constexpr OrdinalType
            forward() const {
                return extent_1_ * extent_2_ * extent_3_;
            }

            KOKKOS_INLINE_FUNCTION constexpr void
            backward(OrdinalType  flattened_index,
                     OrdinalType& extent_1_index,
                     OrdinalType& extent_2_index,
                     OrdinalType& extent_3_index) const {
                extent_1_index = (flattened_index) % extent_1_;
                extent_2_index = (flattened_index / (extent_1_)) % extent_2_;
                extent_3_index = (flattened_index / (extent_1_ * extent_2_)) /* % extent_3_ */;
            }

            template <typename Callable>
            KOKKOS_INLINE_FUNCTION constexpr void
            Call(OrdinalType flattened_index,
                 Callable&&  a_callable) const {
                OrdinalType extent_1_index;
                OrdinalType extent_2_index;
                OrdinalType extent_3_index;
                backward(flattened_index,
                         extent_1_index,
                         extent_2_index,
                         extent_3_index);
                a_callable(extent_1_index,
                           extent_2_index,
                           extent_3_index);
            }

        protected:
            OrdinalType extent_1_;
            OrdinalType extent_2_;
            OrdinalType extent_3_;
        };

    } // namespace utility
} // namespace koliop

#endif
