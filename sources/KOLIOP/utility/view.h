// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_VIEW_H
#define KOLIOP_UTILITY_VIEW_H

#include <Kokkos_Core.hpp>

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
            /// @brief Unmanaged Host layout Left
            /// @tparam Extent
            template <typename Extent>
            using UHL = Kokkos::View<Extent,
                                     Kokkos::LayoutLeft,
                                     Kokkos::DefaultHostExecutionSpace::memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;
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

            /// @brief Unmanaged Device layout Left
            /// May be similar to UHL if the device is the CPU.
            /// @tparam Extent
            template <typename Extent>
            using UDL = Kokkos::View<Extent,
                                     Kokkos::LayoutLeft,
                                     Kokkos::DefaultExecutionSpace::memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;
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

            /// @brief Unmanaged Device layout Default for device
            /// @tparam Extent
            template <typename Extent>
            using UDD = Kokkos::View<Extent,
                                     Kokkos::DefaultExecutionSpace::array_layout,
                                     Kokkos::DefaultExecutionSpace::memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>>;
            template <typename T>
            using UDD1 = UDD<T*>;
            template <typename T>
            using UDD2 = UDD<T**>;
            template <typename T>
            using UDD3 = UDD<T***>;
            template <typename T>
            using UDD4 = UDD<T****>;
            template <typename T>
            using UDD5 = UDD<T*****>;
            template <typename T>
            using UDD6 = UDD<T******>;

            /// @brief Managed Host layout Left
            /// @tparam Extent
            template <typename Extent>
            using MHL = Kokkos::View<Extent,
                                     Kokkos::LayoutLeft,
                                     Kokkos::DefaultHostExecutionSpace::memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Restrict>>;
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

            /// @brief Managed Device layout Left
            /// @tparam Extent
            template <typename Extent>
            using MDL = Kokkos::View<Extent,
                                     Kokkos::LayoutLeft,
                                     Kokkos::DefaultExecutionSpace::memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Restrict>>;
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

            /// @brief Managed Device layout Default for device
            /// @tparam Extent
            template <typename Extent>
            using MDD = Kokkos::View<Extent,
                                     Kokkos::DefaultExecutionSpace::array_layout,
                                     Kokkos::DefaultExecutionSpace::memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Restrict>>;
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


#if defined(KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
            // TODO(Etienne M): Make sure the copies are elided (that is, Gysela
            // validates)
            template <typename Extent>
            using ExternalInputL = UDL<Extent>;
            template <typename Extent>
            using InternalInputL = UDL<Extent>;
#else
            // The inputs MUST come from the host memory.

            template <typename Extent>
            using ExternalInputL = UHL<Extent>;
            template <typename Extent>
            using InternalInputL = MDL<Extent>;
#endif

            template <typename T>
            using ExternalInputL1 = ExternalInputL<T*>;
            template <typename T>
            using ExternalInputL2 = ExternalInputL<T**>;
            template <typename T>
            using ExternalInputL3 = ExternalInputL<T***>;
            template <typename T>
            using ExternalInputL4 = ExternalInputL<T****>;
            template <typename T>
            using ExternalInputL5 = ExternalInputL<T*****>;
            template <typename T>
            using ExternalInputL6 = ExternalInputL<T******>;

            template <typename T>
            using InternalInputL1 = InternalInputL<T*>;
            template <typename T>
            using InternalInputL2 = InternalInputL<T**>;
            template <typename T>
            using InternalInputL3 = InternalInputL<T***>;
            template <typename T>
            using InternalInputL4 = InternalInputL<T****>;
            template <typename T>
            using InternalInputL5 = InternalInputL<T*****>;
            template <typename T>
            using InternalInputL6 = InternalInputL<T******>;
        };

    } // namespace utility
} // namespace koliop

#endif
