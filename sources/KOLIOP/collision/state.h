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
                          hat_Zs_extent_0}
                // TODO(Etienne M): Compile time code based on Npolmax_ value (small)
                , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, // Initialize to zero
                                          "C_j_"),
                       Npolmax_,
                       Npolmax_}
                , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                           "C_jl_"),
                        index_max_,
                        Npolmax_,
                        Npolmax_} {

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

                PrepareCj_Cjl(C_j_, C_jl_);
            }

        protected:
            template <typename DeviceView0,
                      typename DeviceView1>
            static void PrepareCj_Cjl(DeviceView0&& C_j,
                                      DeviceView1&& C_jl) {
                // a_host_visible_C_j is zero initialized.
                auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

                const auto the_polynomial_degree = C_j.extent(0);

                // TODO(Etienne M): what does Cj looks like.

                for(std::size_t i = 0; i < the_polynomial_degree; ++i) {
                    a_host_visible_C_j(0, i) = 1.0;
                    a_host_visible_C_j(1, i) = static_cast<Flo64>(-i);
                }

                if(the_polynomial_degree > 2) {
                    a_host_visible_C_j(2, 2) = 0.5;
                    if(the_polynomial_degree > 3) {
                        a_host_visible_C_j(2, 3) = 1.5;
                        a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                    }
                }

                koliop::utility::SmartDeepCopy(C_j, a_host_visible_C_j);

                PrepareCjl(a_host_visible_C_j, C_jl);
            }

            template <typename DeviceView,
                      typename HostView>
            static void PrepareCjl(const HostView& a_host_visible_C_j,
                                   DeviceView&&    C_jl) {
                // a_host_visible_C_jl is zero initialized.
                auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl);

                const auto the_polynomial_degree = a_host_visible_C_j.extent(0);

                const auto Stuff = [&](auto j_tmp, auto l_tmp) {
                    for(std::size_t i = 0; i < (j_tmp + l_tmp); ++i) {
                        a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                    }

                    for(std::size_t alpha = 0; alpha < j_tmp; ++alpha) {
                        for(std::size_t beta = 0; beta < l_tmp; ++beta) {
                            a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) = a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) +
                                                                              a_host_visible_C_j(alpha, j_tmp) * a_host_visible_C_j(beta, l_tmp);
                        }
                    }
                };

                for(std::size_t i = 0; i < the_polynomial_degree; ++i) {
                    for(std::size_t j = 0; j < the_polynomial_degree; ++j) {
                        Stuff(i, j);
                    }
                }

                koliop::utility::SmartDeepCopy(C_jl, a_host_visible_C_jl);
            }

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
            Kokkos::View<Flo64**>                     C_j_;
            Kokkos::View<Flo64***>                    C_jl_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
