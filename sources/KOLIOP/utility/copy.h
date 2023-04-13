// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_COPY_H
#define KOLIOP_UTILITY_COPY_H

#include <Kokkos_Core.hpp>

namespace koliop {
    namespace utility {

        ////////////////////////////////////////////////////////////////////////
        // Copy definition
        ////////////////////////////////////////////////////////////////////////

        /// TODO(Etienne M):
        /// Add execution space aware version of the functions below ?
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
            template <typename SourceView,
                      typename DestinationView>
            static auto
            Shallow(SourceView&&      a_source_view,
                    DestinationView&& a_destination_view);

            /// The storage for the a_destination_view should already exist and the extents
            /// must match.
            ///
            template <typename SourceView,
                      typename DestinationView>
            static auto
            Deep(SourceView&&      a_source_view,
                 DestinationView&& a_destination_view);

            /// May or may not allocate new memory to represent the_raw_data. As
            /// such it is required that the_raw_data's lifetime may be as long
            /// as the one of the returned view.
            ///
            template <typename UnmanagedView,
                      template <typename>
                      typename DestinationViewTemplate,
                      typename... Indicies>
            static auto
            RawDataToView(typename UnmanagedView::value_type* the_raw_data,
                          const char*                         the_view_label,
                          Indicies... the_indicies);
        };


        ////////////////////////////////////////////////////////////////////////
        // Copy methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename SourceView,
                  typename DestinationView>
        auto Copy::Shallow(SourceView&&      a_source_view,
                           DestinationView&& a_destination_view) {
            if constexpr(std::is_same_v<typename std::remove_reference_t<SourceView>::memory_space,
                                        typename std::remove_reference_t<DestinationView>::memory_space> &&
                         std::is_same_v<typename std::remove_reference_t<SourceView>::array_layout,
                                        typename std::remove_reference_t<DestinationView>::array_layout>) {
                a_destination_view = a_source_view;
            } else {
                Deep(a_source_view, a_destination_view);
            }

            return a_destination_view;
        }

        template <typename SourceView,
                  typename DestinationView>
        auto Copy::Deep(SourceView&&      a_source_view,
                        DestinationView&& a_destination_view) {
            if constexpr(std::is_same_v<typename std::remove_reference_t<SourceView>::memory_space,
                                        typename std::remove_reference_t<DestinationView>::memory_space> ||
                         std::is_same_v<typename std::remove_reference_t<SourceView>::array_layout,
                                        typename std::remove_reference_t<DestinationView>::array_layout>) {
                // Either we are in the same memory space, or we have a
                // compatible layout which makes copying a simple memcpy.
                Kokkos::deep_copy(a_destination_view, a_source_view);
            } else {

                // // TODO(Etienne M): create_mirror_view_and_copy ?
                // auto a_host_visible_source_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                //                                                                                            Kokkos::WithoutInitializing),
                //                                                                         a_source_view);

                // TODO(Etienne M): If we pass a memory_space to
                // create_mirror_view or create_mirror_view_and_copy, we should
                // be able to directly copy from memory space A to memory space
                // B. Then a simple shallow copy assignment to
                // a_destination_view and we are good.

                auto a_host_visible_source_mirror = Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::WithoutInitializing), a_source_view);
                auto a_host_visible_target_mirror = Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::WithoutInitializing), a_destination_view);

                // Could be a no-op.
                Kokkos::deep_copy(a_host_visible_source_mirror,
                                  a_source_view);

                // Resolve potential layout issues on the host.
                Kokkos::deep_copy(a_host_visible_target_mirror,
                                  a_host_visible_source_mirror);

                // Could be a no-op.
                Kokkos::deep_copy(a_destination_view,
                                  a_host_visible_target_mirror);
            }

            return a_destination_view;
        }

        template <typename UnmanagedView,
                  template <typename>
                  typename DestinationViewTemplate,
                  typename... Indicies>
        auto Copy::RawDataToView(typename UnmanagedView::value_type* the_raw_data,
                                 const char*                         the_view_label,
                                 Indicies... the_indicies) {

            // NOTE:
            // Assuming host as device, given a constant host ptr, we do not copy and do a                       simple assignment into a const view
            // Assuming gpu  as device, given a constant host ptr, we do     copy into a non const view and do a simple assignment into a const view
            //
            // By const view, we mean a const value type.

            using DestinationView = DestinationViewTemplate<std::remove_const_t<typename UnmanagedView::value_type>>;

            if constexpr(std::is_same_v<typename std::remove_reference_t<UnmanagedView>::array_layout,
                                        typename std::remove_reference_t<DestinationView>::array_layout> &&
                         std::is_same_v<typename std::remove_reference_t<UnmanagedView>::memory_space,
                                        typename std::remove_reference_t<DestinationView>::memory_space>) {
                return UnmanagedView{the_raw_data,
                                     the_indicies...};
            } else {
                return Deep(UnmanagedView{the_raw_data,
                                          the_indicies...},
                            DestinationView{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                               std::string{the_view_label}),
                                            the_indicies...});
            }
        }

    } // namespace utility
} // namespace koliop

#endif
