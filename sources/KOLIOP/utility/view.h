// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_VIEW_H
#define KOLIOP_UTILITY_VIEW_H

#include <Kokkos_Core.hpp>

namespace koliop {
    namespace utility {

        // /// TODO(Etienne M): what about giving an execution space ?
        // /// NOTE: Kokkos::view_alloc seems broken in our case.
        // ///
        // template <typename Dimension,
        //           typename Layout,
        //           typename MemorySpace,
        //           typename T,
        //           typename... Extents>
        // auto CreateUnmanagedView(const char* an_identifier,
        //                          T*          a_pointer,
        //                          Extents... the_extents) {
        //     using View = Kokkos::View<Dimension,
        //                               Layout,
        //                               MemorySpace,
        //                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        //     // NOTE: Completely broken view_alloc ?
        //     return View{// Kokkos::view_alloc(an_execution_space,
        //                 //                    Kokkos::WithoutInitializing,
        //                 //                    an_identifier,
        //                 //                    a_pointer),
        //                 a_pointer,
        //                 the_extents...};
        // }

        // /// TODO(Etienne M): To be refined
        // ///
        // template <typename View0,
        //           typename View1>
        // void SmartCopy(const HostView2DDynamic& comb_mat,
        //                DeviceView2DDynamic&     comb_mat_) {
        //     {
        //         // Same layout? | Input ptr on: | Output ptr on:      |
        //         //              | Host | Other  | Host | (same) Other |
        //         // Host to host
        //         //          Yes |    1 | 0      | 1    | 0            | No op (reuse) or alloc+memcpy
        //         //           No |    1 | 0      | 1    | 0            | alloc+transposition/memcpy
        //         // Device to device
        //         //          Yes |    0 | 1      | 0    | 1            | No op (reuse) or alloc+memcpy
        //         //           No |    0 | 1      | 0    | 1            | alloc+transposition/memcpy
        //         // Host to device
        //         //          Yes |    1 | 0      | 0    | 1            | alloc+memcpy
        //         //           No |    1 | 0      | 0    | 1            | Transpose on host, then memcpy
        //         // Device to host
        //         //          Yes |    0 | 1      | 1    | 0            | alloc+memcpy
        //         //           No |    0 | 1      | 1    | 0            | alloc+transposition/memcpy

        //         if constexpr(std::is_same<HostView2DDynamic::array_layout,
        //                                   DeviceView2DDynamic::array_layout>::value)

        //             auto tmp = Kokkos::create_mirror_view(comb_mat_);
        //         Kokkos::deep_copy(tmp, comb_mat);
        //         Kokkos::deep_copy(comb_mat_, tmp);
        //     }
        // }

        /// May still do a deep copy if or memory spaces layouts are not
        /// compatible.
        ///
        template <typename FromView,
                  typename ToView>
        void SmartShallowCopy(FromView&& from_view,
                              ToView&&   to_view) {
            if constexpr(std::is_same_v<typename std::remove_reference_t<FromView>::memory_space,
                                        typename std::remove_reference_t<ToView>::memory_space>) {
                to_view = from_view;
            } else {
                SmartDeepCopy(from_view, to_view);
            }
        }

        template <typename FromView,
                  typename ToView>
        void SmartDeepCopy(FromView&& from_view,
                           ToView&&   to_view) {
            if constexpr(std::is_same_v<typename std::remove_reference_t<FromView>::memory_space,
                                        typename std::remove_reference_t<ToView>::memory_space> ||
                         std::is_same_v<typename std::remove_reference_t<FromView>::array_layout,
                                        typename std::remove_reference_t<ToView>::array_layout>) {
                // Either we are in the same memory space, or we have a
                // compatible layout which makes copying a simple memcpy.
                Kokkos::deep_copy(to_view, from_view);
            } else {
                // TODO(Etienne M): Do we really need to copy to the host ?
                auto a_host_visible_source_mirror = Kokkos::create_mirror_view(from_view);
                auto a_host_visible_target_mirror = Kokkos::create_mirror_view(to_view);

                // Could be a no-op.
                Kokkos::deep_copy(a_host_visible_source_mirror,
                                  from_view);

                // Resolve potential layout issues on the host.
                Kokkos::deep_copy(a_host_visible_target_mirror,
                                  a_host_visible_source_mirror);

                // Could be a no-op.
                Kokkos::deep_copy(to_view,
                                  a_host_visible_target_mirror);
            }
        }

    } // namespace utility
} // namespace koliop

#endif
