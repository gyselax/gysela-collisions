// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_VIEW_H
#define KOLIOP_UTILITY_VIEW_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace utility {

        ////////////////////////////////////////////////////////////////////////
        // View definition
        ////////////////////////////////////////////////////////////////////////

        /// View[U|M]               [H|D]                              [L|R|D]                [1-8]
        ///      Unmanaged/Managed | Host or Default (device or host) | Left or Right layout | Dimension
        ///
        /// NOTE: All the views a using the Kokkos::Restrict MemoryTraits. This
        /// means that one should not try to access a given value via more than
        /// one pointer alias.
        ///
        struct View {
        public:
            /// @brief Unmanaged Device layout Left
            /// May be similar to UHL if the device is the CPU.
            /// @tparam Extent
            template <typename T, typename Extents>
            using UDL = Kokkos::View<T,
                                     Extents,
                                     Kokkos::layout_left,
                                     Kokkos::Experimental::Accessor<T,
                                                                    Kokkos::DefaultExecutionSpace::memory_space,
                                                                    Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>>;
            template <typename T>
            using UDL1 = UDL<T, Kokkos::dextents<OrdinalType, 1>>;
            template <typename T>
            using UDL2 = UDL<T, Kokkos::dextents<OrdinalType, 2>>;
            template <typename T>
            using UDL3 = UDL<T, Kokkos::dextents<OrdinalType, 3>>;
            template <typename T>
            using UDL4 = UDL<T, Kokkos::dextents<OrdinalType, 4>>;
            template <typename T>
            using UDL5 = UDL<T, Kokkos::dextents<OrdinalType, 5>>;
            template <typename T>
            using UDL6 = UDL<T, Kokkos::dextents<OrdinalType, 6>>;

            /// @brief Managed Host layout Left
            /// @tparam Extent
            template <typename T, typename Extents>
            using MHL = Kokkos::View<T,
                                     Extents,
                                     Kokkos::layout_left,
                                     Kokkos::Experimental::Accessor<T,
                                                                    Kokkos::DefaultHostExecutionSpace::memory_space,
                                                                    Kokkos::MemoryTraits<Kokkos::Restrict>>>;
            template <typename T>
            using MHL1 = MHL<T, Kokkos::dextents<OrdinalType, 1>>;
            template <typename T>
            using MHL2 = MHL<T, Kokkos::dextents<OrdinalType, 2>>;
            template <typename T>
            using MHL3 = MHL<T, Kokkos::dextents<OrdinalType, 3>>;
            template <typename T>
            using MHL4 = MHL<T, Kokkos::dextents<OrdinalType, 4>>;
            template <typename T>
            using MHL5 = MHL<T, Kokkos::dextents<OrdinalType, 5>>;
            template <typename T>
            using MHL6 = MHL<T, Kokkos::dextents<OrdinalType, 6>>;

            /// @brief Managed Device layout Left
            /// @tparam Extent
            template <typename T, typename Extents>
            using MDL = Kokkos::View<T,
                                     Extents,
                                     Kokkos::layout_left,
                                     Kokkos::Experimental::Accessor<T,
                                                                    Kokkos::DefaultExecutionSpace::memory_space,
                                                                    Kokkos::MemoryTraits<Kokkos::Restrict>>>;
            template <typename T>
            using MDL1 = MDL<T, Kokkos::dextents<OrdinalType, 1>>;
            template <typename T>
            using MDL2 = MDL<T, Kokkos::dextents<OrdinalType, 2>>;
            template <typename T>
            using MDL3 = MDL<T, Kokkos::dextents<OrdinalType, 3>>;
            template <typename T>
            using MDL4 = MDL<T, Kokkos::dextents<OrdinalType, 4>>;
            template <typename T>
            using MDL5 = MDL<T, Kokkos::dextents<OrdinalType, 5>>;
            template <typename T>
            using MDL6 = MDL<T, Kokkos::dextents<OrdinalType, 6>>;
        };

    } // namespace utility
} // namespace koliop

#endif
