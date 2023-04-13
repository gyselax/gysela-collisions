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
            template <typename Float>
            using UHL1 = UHL<Float*>;
            template <typename Float>
            using UHL2 = UHL<Float**>;
            template <typename Float>
            using UHL3 = UHL<Float***>;
            template <typename Float>
            using UHL4 = UHL<Float****>;
            template <typename Float>
            using UHL5 = UHL<Float*****>;
            template <typename Float>
            using UHL6 = UHL<Float******>;

            template <typename Extent>
            using MDL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
            template <typename Float>
            using MDL1 = MDL<Float*>;
            template <typename Float>
            using MDL2 = MDL<Float**>;
            template <typename Float>
            using MDL3 = MDL<Float***>;
            template <typename Float>
            using MDL4 = MDL<Float****>;
            template <typename Float>
            using MDL5 = MDL<Float*****>;
            template <typename Float>
            using MDL6 = MDL<Float******>;

            template <typename Extent>
            using MDD = Kokkos::View<Extent, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
            template <typename Float>
            using MDD1 = MDD<Float*>;
            template <typename Float>
            using MDD2 = MDD<Float**>;
            template <typename Float>
            using MDD3 = MDD<Float***>;
            template <typename Float>
            using MDD4 = MDD<Float****>;
            template <typename Float>
            using MDD5 = MDD<Float*****>;
            template <typename Float>
            using MDD6 = MDD<Float******>;

            template <typename Extent>
            using MHL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space>;
            template <typename Float>
            using MHL1 = MHL<Float*>;
        };

    } // namespace utility
} // namespace koliop

#endif
