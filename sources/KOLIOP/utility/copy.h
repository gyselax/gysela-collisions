// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_COPY_H
#define KOLIOP_UTILITY_COPY_H

#include <Kokkos_Core.hpp>

namespace koliop {
    namespace utility {

        ////////////////////////////////////////////////////////////////////////
        // Copy definition
        ////////////////////////////////////////////////////////////////////////

        ///
        ///
        struct Copy {
        public:
            /// May still do a deep copy if memory spaces or layouts are not
            /// compatible.
            ///
            /// Same layout? | Input ptr on: | Output ptr on:      |
            ///              | Host | Other  | Host | (same) Other |
            /// Host to host
            ///          Yes |    1 | 0      | 1    | 0            | No op (reuse) or alloc+memcpy
            ///           No |    1 | 0      | 1    | 0            | alloc+transposition/memcpy
            /// Device to device
            ///          Yes |    0 | 1      | 0    | 1            | No op (reuse) or alloc+memcpy
            ///           No |    0 | 1      | 0    | 1            | alloc+transposition/memcpy
            /// Host to device
            ///          Yes |    1 | 0      | 0    | 1            | alloc+memcpy
            ///           No |    1 | 0      | 0    | 1            | Transpose on host, then memcpy
            /// Device to host
            ///          Yes |    0 | 1      | 1    | 0            | alloc+memcpy
            ///           No |    0 | 1      | 1    | 0            | alloc+transposition/memcpy
            ///
            /// NOTE: In case a DeepCopy is realized, the extents must match and
            /// the storage must be allocated. TODO(Etienne M): In this case,
            /// should be resize in Shallow() before calling Deep ?
            /// NOTE: In case the shallow copy is realized, we loose the View's
            /// labe !!
            ///
            template <typename FromView,
                      typename ToView>
            static void
            Shallow(FromView&& from_view,
                    ToView&&   to_view);

            /// The storage for the to_view should already exist and the extents
            /// must match.
            ///
            template <typename FromView,
                      typename ToView>
            static void
            Deep(FromView&& from_view,
                 ToView&&   to_view);
        };


        ////////////////////////////////////////////////////////////////////////
        // Copy methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename FromView,
                  typename ToView>
        void Copy::Shallow(FromView&& from_view,
                           ToView&&   to_view) {
            if constexpr(std::is_same_v<typename std::remove_reference_t<FromView>::memory_space,
                                        typename std::remove_reference_t<ToView>::memory_space> &&
                         std::is_same_v<typename std::remove_reference_t<FromView>::array_layout,
                                        typename std::remove_reference_t<ToView>::array_layout>) {
                to_view = from_view;
            } else {
                Deep(from_view, to_view);
            }
        }

        template <typename FromView,
                  typename ToView>
        void Copy::Deep(FromView&& from_view,
                        ToView&&   to_view) {
            if constexpr(std::is_same_v<typename std::remove_reference_t<FromView>::memory_space,
                                        typename std::remove_reference_t<ToView>::memory_space> ||
                         std::is_same_v<typename std::remove_reference_t<FromView>::array_layout,
                                        typename std::remove_reference_t<ToView>::array_layout>) {
                // Either we are in the same memory space, or we have a
                // compatible layout which makes copying a simple memcpy.
                Kokkos::deep_copy(to_view, from_view);
            } else {

                // TODO(Etienne M): create_mirror_view_and_copy ?

                // TODO(Etienne M): Do we really need to copy to the host ?
                auto a_host_visible_source_mirror = Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::WithoutInitializing), from_view);
                auto a_host_visible_target_mirror = Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::WithoutInitializing), to_view);

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
