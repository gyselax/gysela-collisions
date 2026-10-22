// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_ITERATE_H
#define KOLIOP_UTILITY_ITERATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/utility/flatten.h"

namespace koliop {
    namespace utility {

        struct IntraTeam {
        public:
            template <typename OrdinalType,
                      typename TeamHandle,
                      typename Callable>
            static KOKKOS_INLINE_FUNCTION void
            Iterate2D(OrdinalType extent_1,
                      OrdinalType extent_2,
                      TeamHandle  a_team_handle,
                      Callable&&  a_callable) {
#if defined(KOLIOP_BACKEND_IS_GPU_LIKE)
                const koliop::utility::Flatten2D flattened_range{extent_1, extent_2};
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(a_team_handle,
                                            flattened_range.forward()),
                    [&](auto flattened_index) {
                        flattened_range.Call(flattened_index, a_callable);
                    });
#else
                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                              std::remove_reference_t<decltype(a_team_handle)> >(a_team_handle,
                                                                                                 extent_1, extent_2),
                    a_callable);
#endif
            }
        };

    } // namespace utility
} // namespace koliop

#endif
