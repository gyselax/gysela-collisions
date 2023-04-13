// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_VIEW_H
#define KOLIOP_UTILITY_VIEW_H

#include <Kokkos_Core.hpp>

namespace koliop {
    namespace utility {

        ////////////////////////////////////////////////////////////////////////
        // View definition
        ////////////////////////////////////////////////////////////////////////

        /// TODO(Etienne M): Cleanup that stuff.
        /// View[U|M][H|D][L|R|D][1-9]
        /// Unmanaged/Managed | Host or Default (device or host) | Left or Right layout | Dimension
        ///
        struct View {
        public:
            template <typename Extent>
            using UHL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            template <typename T>
            using UHL1 = UHL<T*>;
            template <typename T>
            using UHL2 = UHL<T**>;
            template <typename T>
            using UHL3 = UHL<T***>;
            template <typename T>
            using UHL4 = UHL<T****>;
            template <typename T>
            using UHL5 = UHL<T*****>;
            template <typename T>
            using UHL6 = UHL<T******>;

            template <typename Extent>
            using MDL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
            template <typename T>
            using MDL1 = MDL<T*>;
            template <typename T>
            using MDL2 = MDL<T**>;
            template <typename T>
            using MDL3 = MDL<T***>;
            template <typename T>
            using MDL4 = MDL<T****>;
            template <typename T>
            using MDL5 = MDL<T*****>;
            template <typename T>
            using MDL6 = MDL<T******>;

            template <typename Extent>
            using MDD = Kokkos::View<Extent, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
            template <typename T>
            using MDD1 = MDD<T*>;
            template <typename T>
            using MDD2 = MDD<T**>;
            template <typename T>
            using MDD3 = MDD<T***>;
            template <typename T>
            using MDD4 = MDD<T****>;
            template <typename T>
            using MDD5 = MDD<T*****>;
            template <typename T>
            using MDD6 = MDD<T******>;

            template <typename Extent>
            using MHL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space>;
            template <typename T>
            using MHL1 = MHL<T*>;
        };

    } // namespace utility
} // namespace koliop

#endif
