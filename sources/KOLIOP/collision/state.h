// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // State definition
        ////////////////////////////////////////////////////////////////////////


        /// Execution Spaces:
        ///     Where you physically execute
        ///     Memory Spaces:
        ///         Where you physically store
        ///         Memory Layout:
        ///             How to index
        ///         Memory Traits:
        ///             Modifier: stream, random, atomic, texture memory
        /// Execution Patterns:
        ///     Parallel for/scan/reduce
        /// Execution Policies:
        ///     Range, team, dynamic or static scheduling
        ///
        template <typename Policy>
        class State {
        public:
            using PolicyType = Policy;

        public:
            State(std::size_t Npolmax,
                  std::size_t index_max,
                  std::size_t Nvpar,
                  std::size_t Nspecies,
                  std::size_t pSpecies_id,
                  std::size_t mu_id,
                  bool        collision_interspecies,

                  const Flo64* comb_mat,
                  std::size_t  comb_mat_extent_0,
                  std::size_t  comb_mat_extent_1,

                  const Flo64* hat_As,
                  std::size_t  hat_As_extent_0,

                  const Flo64* hat_Zs,
                  std::size_t  hat_Zs_extent_0)
                : Npolmax_{Npolmax}
                , index_max_{index_max}
                , Nvpar_{Nvpar}
                , Nspecies_{Nspecies}
                , pSpecies_id_{pSpecies_id}
                , mu_id_{mu_id}
                , collision_interspecies_{collision_interspecies}

                , comb_mat_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "comb_mat_"),
                            comb_mat_extent_0,
                            comb_mat_extent_1}
                , hat_As_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "hat_As_"),
                          hat_As_extent_0}
                , hat_Zs_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "hat_Zs_"),
                          hat_Zs_extent_0} {

                auto unmanaged_comb_mat = Kokkos::View<const Flo64**,
                                                       Kokkos::LayoutLeft,
                                                       Kokkos::HostSpace,
                                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_comb_mat",
                                                                                                comb_mat,
                                                                                                comb_mat_extent_0,
                                                                                                comb_mat_extent_1};
                // TODO(Etienne M): Deep or shallow ? On CPU, shallow would be
                // enough.
                koliop::utility::SmartDeepCopy(unmanaged_comb_mat, comb_mat_);

                auto unmanaged_hat_As = Kokkos::View<const Flo64*,
                                                     Kokkos::HostSpace,
                                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_hat_As",
                                                                                              hat_As,
                                                                                              hat_As_extent_0};
                koliop::utility::SmartDeepCopy(unmanaged_hat_As, hat_As_);

                auto unmanaged_hat_Zs = Kokkos::View<const Flo64*,
                                                     Kokkos::HostSpace,
                                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_hat_Zs",
                                                                                              hat_Zs,
                                                                                              hat_Zs_extent_0};
                koliop::utility::SmartDeepCopy(unmanaged_hat_Zs, hat_Zs_);
            }

        protected:
            std::size_t Npolmax_;
            std::size_t index_max_;
            std::size_t Nvpar_;
            std::size_t Nspecies_;
            std::size_t pSpecies_id_;
            std::size_t mu_id_;
            bool        collision_interspecies_;

            Kokkos::View<Flo64**, Kokkos::LayoutLeft> comb_mat_;
            Kokkos::View<Flo64*>                      hat_As_;
            Kokkos::View<Flo64*>                      hat_Zs_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
