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
                  std::size_t Nmu,
                  std::size_t Nvpar,
                  std::size_t Nspecies,
                  std::size_t pSpecies_id,
                  std::size_t mu_id,
                  std::size_t istart,
                  std::size_t iend,
                  std::size_t jstart,
                  std::size_t jend,
                  std::size_t collstart,
                  std::size_t collend,
                  std::size_t R0,
                  bool        collision_interspecies,
                  bool        SOL,
                  bool        LIM,

                  const Flo64* comb_mat,
                  std::size_t  comb_mat_extent_0,
                  std::size_t  comb_mat_extent_1,

                  const Flo64* hat_As,
                  std::size_t  hat_As_extent_0,

                  const Flo64* hat_Zs,
                  std::size_t  hat_Zs_extent_0

                  // const Flo64* mug,
                  // std::size_t  mug_extent_0,

                  // const Flo64* vparg,
                  // std::size_t  vparg_extent_0,

                  // const Flo64* coeff_intdmu,
                  // std::size_t  coeff_intdmu_extent_0,

                  // const Flo64* coeff_intdvpar,
                  // std::size_t  coeff_intdvpar_extent_0
                  )
                : Npolmax_{Npolmax}
                , index_max_{index_max}
                , Nmu_{Nmu}
                , Nvpar_{Nvpar}
                , Nspecies_{Nspecies}
                , pSpecies_id_{pSpecies_id}
                , mu_id_{mu_id}
                , istart_{istart}
                , iend_{iend}
                , jstart_{jstart}
                , jend_{jend}
                , collstart_{collstart}
                , collend_{collend}
                , R0_{R0}
                , collision_interspecies_{collision_interspecies}
                , SOL_{SOL}
                , LIM_{LIM}
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
                // , mug_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                //                           "mug_"),
                //        mug_extent_0}
                // , vparg_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                //                             "vparg_"),
                //          vparg_extent_0}
                // , coeff_intdmu_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                //                                    "coeff_intdmu_"),
                //                 coeff_intdmu_extent_0}
                // , coeff_intdvpar_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                //                                      "coeff_intdvpar_"),
                //                   coeff_intdvpar_extent_0}

                // TODO(Etienne M): Compile time code based on Npolmax_ value (small)
                , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, // Initialize to zero
                                          "C_j_"),
                       Npolmax_,
                       Npolmax_}
                , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                           "C_jl_"),
                        index_max_ + 1, // TODO(Etienne M): INDEX Fortran s***
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
                                                     Kokkos::LayoutLeft,
                                                     Kokkos::HostSpace,
                                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_hat_As",
                                                                                              hat_As,
                                                                                              hat_As_extent_0};
                koliop::utility::SmartDeepCopy(unmanaged_hat_As, hat_As_);

                auto unmanaged_hat_Zs = Kokkos::View<const Flo64*,
                                                     Kokkos::LayoutLeft,
                                                     Kokkos::HostSpace,
                                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_hat_Zs",
                                                                                              hat_Zs,
                                                                                              hat_Zs_extent_0};
                koliop::utility::SmartDeepCopy(unmanaged_hat_Zs, hat_Zs_);

                // auto unmanaged_mug = Kokkos::View<const Flo64*,
                //                                   Kokkos::LayoutLeft,
                //                                   Kokkos::HostSpace,
                //                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_mug",
                //                                                                            mug,
                //                                                                            mug_extent_0};
                // koliop::utility::SmartDeepCopy(unmanaged_mug, mug_);

                // auto unmanaged_vparg = Kokkos::View<const Flo64*,
                //                                     Kokkos::LayoutLeft,
                //                                     Kokkos::HostSpace,
                //                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_vparg",
                //                                                                              vparg,
                //                                                                              vparg_extent_0};
                // koliop::utility::SmartDeepCopy(unmanaged_vparg, vparg_);

                // auto unmanaged_coeff_intdmu = Kokkos::View<const Flo64*,
                //                                            Kokkos::LayoutLeft,
                //                                            Kokkos::HostSpace,
                //                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_coeff_intdmu",
                //                                                                                     coeff_intdmu,
                //                                                                                     coeff_intdmu_extent_0};
                // koliop::utility::SmartDeepCopy(unmanaged_coeff_intdmu, coeff_intdmu_);

                // auto unmanaged_coeff_intdvpar = Kokkos::View<const Flo64*,
                //                                              Kokkos::LayoutLeft,
                //                                              Kokkos::HostSpace,
                //                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_coeff_intdvpar",
                //                                                                                       coeff_intdvpar,
                //                                                                                       coeff_intdvpar_extent_0};
                // koliop::utility::SmartDeepCopy(unmanaged_coeff_intdvpar, coeff_intdvpar_);

                PrepareCj_Cjl(C_j_, C_jl_);
            }

        protected:
            template <typename DeviceView0,
                      typename DeviceView1>
            static void PrepareCj_Cjl(DeviceView0&& C_j,
                                      DeviceView1&& C_jl) {
                // a_host_visible_C_j is zero initialized.
                auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

                const auto the_polynomial_degree = C_j.extent(1);

                for(std::size_t j = 0; j < the_polynomial_degree; ++j) {
                    a_host_visible_C_j(0, j) = 1.0;
                    a_host_visible_C_j(1, j) = static_cast<Flo64>(-static_cast<int64_t>(j));
                }

                if(the_polynomial_degree > 2) {
                    a_host_visible_C_j(2, 2) = 0.5;
                    if(the_polynomial_degree > 3) {
                        a_host_visible_C_j(2, 3) = 1.5;
                        a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                    }
                }

                koliop::utility::SmartDeepCopy(a_host_visible_C_j, C_j);

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

                koliop::utility::SmartDeepCopy(a_host_visible_C_jl, C_jl);
            }

            std::size_t Npolmax_;
            std::size_t index_max_; // TODO(Etienne M): INDEX Fortran s***
            std::size_t Nmu_;
            std::size_t Nvpar_;
            std::size_t Nspecies_;
            std::size_t pSpecies_id_;
            std::size_t mu_id_;
            std::size_t istart_;
            std::size_t iend_;
            std::size_t jstart_;
            std::size_t jend_;
            std::size_t collstart_;
            std::size_t collend_;
            std::size_t R0_;
            bool        collision_interspecies_;
            bool        SOL_;
            bool        LIM_;

            Kokkos::View<Flo64**> comb_mat_;
            Kokkos::View<Flo64*>  hat_As_;
            Kokkos::View<Flo64*>  hat_Zs_;
            // Kokkos::View<Flo64*>  mug_;
            // Kokkos::View<Flo64*>  vparg_;
            // Kokkos::View<Flo64*>  coeff_intdmu_;
            // Kokkos::View<Flo64*>  coeff_intdvpar_;

            Kokkos::View<Flo64**>  C_j_;
            Kokkos::View<Flo64***> C_jl_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
