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
            State(uint64_t                                                     Npolmax,
                  uint64_t                                                     index_max,
                  uint64_t                                                     Nmu,
                  uint64_t                                                     Nvpar,
                  uint64_t                                                     Nspecies,
                  uint64_t                                                     pSpecies_id,
                  uint64_t                                                     mu_id,
                  uint64_t                                                     istart,
                  uint64_t                                                     iend,
                  uint64_t                                                     jstart,
                  uint64_t                                                     jend,
                  int64_t                                                      collstart,
                  int64_t                                                      collend,
                  Flo64                                                        R0,
                  bool                                                         collision_interspecies,
                  bool                                                         SOL,
                  bool                                                         LIM,
                  const Kokkos::View<const Flo64**,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_comb_mat,
                  const Kokkos::View<const Flo64*,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_hat_As,
                  const Kokkos::View<const Flo64*,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_hat_Zs,
                  const Kokkos::View<const Flo64*,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_mug,
                  const Kokkos::View<const Flo64*,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_vparg,
                  const Kokkos::View<const Flo64*,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_coeff_intdmu,
                  const Kokkos::View<const Flo64*,
                                     Kokkos::LayoutLeft,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>& a_host_visible_coeff_intdvpar)
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
                , comb_mat_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "comb_mat_"), a_host_visible_comb_mat.extent(0), a_host_visible_comb_mat.extent(1)}
                , hat_As_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_As_"), a_host_visible_hat_As.extent(0)}
                , hat_Zs_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_Zs_"), a_host_visible_hat_Zs.extent(0)}
                , mug_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mug_"), a_host_visible_mug.extent(0)}
                , vparg_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "vparg_"), a_host_visible_vparg.extent(0)}
                , coeff_intdmu_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "coeff_intdmu_"), a_host_visible_coeff_intdmu.extent(0)}
                , coeff_intdvpar_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "coeff_intdvpar_"), a_host_visible_coeff_intdvpar.extent(0)}

                // TODO(Etienne M): Compile time code based on Npolmax_ value ?
                , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing,
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

                // TODO(Etienne M): Deep or shallow ? On CPU, shallow would be
                // enough.
                koliop::utility::SmartDeepCopy(a_host_visible_comb_mat, comb_mat_);
                koliop::utility::SmartDeepCopy(a_host_visible_hat_As, hat_As_);
                koliop::utility::SmartDeepCopy(a_host_visible_hat_Zs, hat_Zs_);
                koliop::utility::SmartDeepCopy(a_host_visible_mug, mug_);
                koliop::utility::SmartDeepCopy(a_host_visible_vparg, vparg_);
                koliop::utility::SmartDeepCopy(a_host_visible_coeff_intdmu, coeff_intdmu_);
                koliop::utility::SmartDeepCopy(a_host_visible_coeff_intdvpar, coeff_intdvpar_);

                PrepareCj_Cjl(C_j_, C_jl_);
            }

            void PrepareNuStar(const Flo64* rg,
                               const Flo64* nustaR0_r,
                               const Flo64* q_r,
                               const Flo64* mask_buffer_r,
                               Flo64*       nustar_coeff) const {

                const uint64_t element_offset = istart_;
                const uint64_t element_count  = iend_ + 1 - element_offset;

                rg += element_offset;
                nustaR0_r += element_offset;
                q_r += element_offset;
                mask_buffer_r += element_offset;

                DoPrepareNuStar(Kokkos::View<const Flo64*,
                                             Kokkos::LayoutLeft,
                                             Kokkos::HostSpace,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_rg",
                                                                                      rg, element_count},
                                Kokkos::View<const Flo64*, Kokkos::LayoutLeft, Kokkos::HostSpace,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_nustaR0_r",
                                                                                      nustaR0_r, element_count},
                                Kokkos::View<const Flo64*, Kokkos::LayoutLeft, Kokkos::HostSpace,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_q_r",
                                                                                      q_r, element_count},
                                Kokkos::View<const Flo64*, Kokkos::LayoutLeft, Kokkos::HostSpace,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_mask_buffer_r",
                                                                                      mask_buffer_r, element_count},
                                Kokkos::View<Flo64***, Kokkos::LayoutLeft, Kokkos::HostSpace,
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>{// "unmanaged_host_nustar_coeff",
                                                                                      nustar_coeff, Nspecies_, Nspecies_, element_count});
            }

        protected:
            void DoPrepareNuStar(const Kokkos::View<const Flo64*,
                                                    Kokkos::LayoutLeft,
                                                    Kokkos::HostSpace,
                                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>&                                             a_host_visible_rg,
                                 const Kokkos::View<const Flo64*,
                                                    Kokkos::LayoutLeft,
                                                    Kokkos::HostSpace,
                                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>&                                             a_host_visible_nustaR0_r,
                                 const Kokkos::View<const Flo64*,
                                                    Kokkos::LayoutLeft,
                                                    Kokkos::HostSpace,
                                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>&                                             a_host_visible_q_r,
                                 const Kokkos::View<const Flo64*,
                                                    Kokkos::LayoutLeft,
                                                    Kokkos::HostSpace,
                                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>&                                             a_host_visible_mask_buffer_r,
                                 Kokkos::View<Flo64***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>&& a_host_visible_nustar_coeff) const {

                const uint64_t element_count = a_host_visible_rg.extent(0);

                Kokkos::View<Flo64*, Kokkos::LayoutLeft, Kokkos::HostSpace>
                    mask_space{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                  "mask_space"),
                               element_count};

                if(a_host_visible_q_r(0) < koliop::portability::GyselaConstant<Flo64>::kInfinity) {
                    for(uint64_t i = 0; i < element_count; ++i) {
                        mask_space(i) = 1.0 / a_host_visible_q_r[i];
                    }
                }

                if(!(LIM_ || SOL_)) {
                    for(uint64_t i = 0; i < element_count; ++i) {
                        mask_space(i) = mask_space[i] *
                                        (1.0 - a_host_visible_mask_buffer_r(i));
                    }
                }

                // TODO(Etienne M): Either we keep the pointer to hat_[A|Z]s or
                // we do a deep copy.
                const auto a_host_visible_hat_As = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::HostSpace{},
                                                                                                          Kokkos::WithoutInitializing),
                                                                                       hat_As_);
                const auto a_host_visible_hat_Zs = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::HostSpace{},
                                                                                                          Kokkos::WithoutInitializing),
                                                                                       hat_Zs_);

                // TODO(Etienne M): Obvious loop inter exchange.
                for(uint64_t ispeciesB = 0; ispeciesB < Nspecies_; ++ispeciesB) {
                    for(uint64_t ispeciesA = 0; ispeciesA < Nspecies_; ++ispeciesA) {
                        const Flo64 hat_Aa = a_host_visible_hat_As(ispeciesA);
                        const Flo64 hat_Za = a_host_visible_hat_Zs(ispeciesA);
                        const Flo64 hat_Ab = a_host_visible_hat_As(ispeciesB);
                        const Flo64 hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                        const Flo64 AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                                (1.0 / hat_Aa) *
                                                (1.0 / hat_Aa + 1.0 / hat_Ab);

                        for(uint64_t ir = 0; ir < element_count; ++ir) {
                            // Computation of
                            // AD(r) = sqrt(2) * eps ^ (3 / 2) / (q R0_)*nustaR0_(r)
                            // which is independant on the species considered (eps = r/R0_)
                            const Flo64 eps = a_host_visible_rg(ir) * R0_inverse_;

                            const Flo64 AD = a_host_visible_nustaR0_r(ir) *
                                             mask_space(ir) *
                                             // NOTE:
                                             // sqrt(2)*(rg*R0_inverse_)*sqrt(rg*R0_inverse_)*R0_inverse_ =
                                             // sqrt(2)*pow(r, 5/2)*pow(rg, 3/2)
                                             // sqrt(2)*pow(r, 5/2) is constant here.
                                             koliop::portability::MathematicalConstant<Flo64>::kSQRT2 *
                                             eps *
                                             std::sqrt(eps) *
                                             R0_inverse_;

                            // Computation of nu_ab =
                            // (AD * hat_Za ^ 2 * hat_Zb ^ 2 * Nb) /
                            // (vTa ^ 2 + vTb ^ 2) ^ (3 / 2) * (1 / hat_Aa) *
                            // (1 / hat_Aa + 1 / hat_Ab)
                            a_host_visible_nustar_coeff(ispeciesA, ispeciesB, ir) = AD * AD_factor;
                        }
                    }
                }
            }

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
            Kokkos::View<Flo64*>  mug_;
            Kokkos::View<Flo64*>  vparg_;
            Kokkos::View<Flo64*>  coeff_intdmu_;
            Kokkos::View<Flo64*>  coeff_intdvpar_;

            Kokkos::View<Flo64**>  C_j_;
            Kokkos::View<Flo64***> C_jl_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
