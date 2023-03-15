// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
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
        class State {
        public:
        public:
            State(uint64_t     Npolmax,
                  uint64_t     index_max,
                  uint64_t     Nmu,
                  uint64_t     Nvpar,
                  uint64_t     Nspecies,
                  uint64_t     pSpecies_id,
                  uint64_t     mu_id,
                  uint64_t     istart,
                  uint64_t     iend,
                  uint64_t     jstart,
                  uint64_t     jend,
                  int64_t      collstart,
                  int64_t      collend,
                  Flo64        R0,
                  bool         collision_interspecies,
                  bool         SOL,
                  bool         LIM,
                  const Flo64* comb_mat,
                  uint64_t     comb_mat_extent_0,
                  uint64_t     comb_mat_extent_1,
                  const Flo64* hat_As,
                  uint64_t     hat_As_extent_0,
                  const Flo64* hat_Zs,
                  uint64_t     hat_Zs_extent_0)
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
                , R0_inverse_{1.0 / R0}
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

                // TODO(Etienne M): Compile time code based on Npolmax_ value (small)
                , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, // Initialize to zero
                                          "C_j_"),
                       Npolmax_,
                       Npolmax_}
                , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                           "C_jl_"),
                        index_max_ + 1,
                        Npolmax_,
                        Npolmax_} {

                // This assumption allows for some simpler indexing.
                // TODO(Etienne M): INDEX, recheck all array indexing.
                KOLIOP_ASSERT(istart_ == 0);
                KOLIOP_ASSERT(jstart_ == 0);

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

            void PrepareNuStar(const Flo64* rg,
                               const Flo64* nustaR0_r,
                               const Flo64* q_r,
                               const Flo64* mask_buffer_r,
                               Flo64*       nustar_coeff) {
            }

        protected:
            template <typename DeviceView0,
                      typename DeviceView1>
            static void PrepareCj_Cjl(DeviceView0&& C_j,
                                      DeviceView1&& C_jl) {
                // a_host_visible_C_j is zero initialized.
                auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

                const auto the_polynomial_degree = C_j.extent(1);

                for(uint64_t j = 0; j < the_polynomial_degree; ++j) {
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

                const auto Stuff = [&](auto j_tmp, auto l_tmp) {
                    // for(uint64_t i = 0; i <= (j_tmp + l_tmp); ++i) {
                    //     // Is that necessary ?
                    //     a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                    // }

                    for(uint64_t alpha = 0; alpha <= j_tmp; ++alpha) {
                        for(uint64_t beta = 0; beta <= l_tmp; ++beta) {
                            a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) = a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) +
                                                                              a_host_visible_C_j(alpha, j_tmp) * a_host_visible_C_j(beta, l_tmp);
                        }
                    }
                };

                for(uint64_t i = 0; i < a_host_visible_C_jl.extent(1); ++i) {
                    for(uint64_t j = 0; j < a_host_visible_C_jl.extent(2); ++j) {
                        Stuff(i, j);
                    }
                }

                koliop::utility::SmartDeepCopy(a_host_visible_C_jl, C_jl);
            }

            uint64_t Npolmax_;
            uint64_t index_max_; // TODO(Etienne M): INDEX Fortran s***
            uint64_t Nmu_;
            uint64_t Nvpar_;
            uint64_t Nspecies_;
            uint64_t pSpecies_id_;
            uint64_t mu_id_;
            uint64_t istart_;
            uint64_t iend_; // TODO(Etienne M): INDEX Fortran s***
            uint64_t jstart_;
            uint64_t jend_;
            int64_t  collstart_;
            int64_t  collend_;
            Flo64    R0_inverse_;
            bool     collision_interspecies_;
            bool     SOL_;
            bool     LIM_;

            Kokkos::View<Flo64**> comb_mat_;
            Kokkos::View<Flo64*>  hat_As_;
            Kokkos::View<Flo64*>  hat_Zs_;

            Kokkos::View<Flo64**>  C_j_;
            Kokkos::View<Flo64***> C_jl_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
