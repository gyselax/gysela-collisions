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
        /// View[U|M]               [H|D]                              [L|R|D]                [1-8]
        ///      Unmanaged/Managed | Host or Default (device or host) | Left or Right layout | Dimension
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
            using UDL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            template <typename T>
            using UDL1 = UDL<T*>;
            template <typename T>
            using UDL2 = UDL<T**>;
            template <typename T>
            using UDL3 = UDL<T***>;
            template <typename T>
            using UDL4 = UDL<T****>;
            template <typename T>
            using UDL5 = UDL<T*****>;
            template <typename T>
            using UDL6 = UDL<T******>;

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
            template <typename T>
            using MHL2 = MHL<T**>;
            template <typename T>
            using MHL3 = MHL<T***>;
            template <typename T>
            using MHL4 = MHL<T****>;
            template <typename T>
            using MHL5 = MHL<T*****>;
            template <typename T>
            using MHL6 = MHL<T******>;

#if defined(KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
            // TODO(Etienne M): Make sure the copies are elided (that is, it
            // validates)
            template <typename Extent>
            using ExternalInput = UDL<Extent>;
            template <typename Extent>
            using InternalInput = UDL<Extent>;
#else
            // The inputs MUST come from the host memory.

            template <typename Extent>
            using ExternalInput = UHL<Extent>;
            template <typename Extent>
            using InternalInput = MDL<Extent>;
#endif

            template <typename T>
            using ExternalInput1 = ExternalInput<T*>;
            template <typename T>
            using ExternalInput2 = ExternalInput<T**>;
            template <typename T>
            using ExternalInput3 = ExternalInput<T***>;
            template <typename T>
            using ExternalInput4 = ExternalInput<T****>;
            template <typename T>
            using ExternalInput5 = ExternalInput<T*****>;
            template <typename T>
            using ExternalInput6 = ExternalInput<T******>;

            template <typename T>
            using InternalInput1 = InternalInput<T*>;
            template <typename T>
            using InternalInput2 = InternalInput<T**>;
            template <typename T>
            using InternalInput3 = InternalInput<T***>;
            template <typename T>
            using InternalInput4 = InternalInput<T****>;
            template <typename T>
            using InternalInput5 = InternalInput<T*****>;
            template <typename T>
            using InternalInput6 = InternalInput<T******>;
        };

    } // namespace utility
} // namespace koliop

#endif
