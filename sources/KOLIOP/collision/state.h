// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        // TODO(Etienne M): Cleanup that stuff.
        // View[U|M][H|D][L|R|D][1-9]
        // Unmanaged/Managed | Host or Default (device or host) | Left or Right layout | Dimension
        template <typename Float>
        using ViewUHL1 = Kokkos::View<Float*, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        template <typename Float>
        using ViewUHL2 = Kokkos::View<Float**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        template <typename Float>
        using ViewUHL3 = Kokkos::View<Float***, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        template <typename Float>
        using ViewUHL4 = Kokkos::View<Float****, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        template <typename Float>
        using ViewUHL5 = Kokkos::View<Float*****, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        template <typename Float>
        using ViewUHL6 = Kokkos::View<Float******, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // template <typename Float>
        // using ViewMDL1 = Kokkos::View<Float*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
        // template <typename Float>
        // using ViewMDL2 = Kokkos::View<Float**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
        // template <typename Float>
        // using ViewMDL3 = Kokkos::View<Float***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
        // template <typename Float>
        // using ViewMDL4 = Kokkos::View<Float****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
        // template <typename Float>
        // using ViewMDL5 = Kokkos::View<Float*****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
        // template <typename Float>
        // using ViewMDL6 = Kokkos::View<Float******, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;

        template <typename Float>
        using ViewMDD1 = Kokkos::View<Float*, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDD2 = Kokkos::View<Float**, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDD3 = Kokkos::View<Float***, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDD4 = Kokkos::View<Float****, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDD5 = Kokkos::View<Float*****, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDD6 = Kokkos::View<Float******, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;


        ////////////////////////////////////////////////////////////////////////
        // State definition
        ////////////////////////////////////////////////////////////////////////

        ///
        class State {
        public:
        public:
            State(uint64_t                     Npolmax,
                  uint64_t                     index_max,
                  uint64_t                     Nmu,
                  uint64_t                     Nvpar,
                  uint64_t                     Nspecies,
                  uint64_t                     pSpecies_id,
                  uint64_t                     mu_id,
                  uint64_t                     istart,
                  uint64_t                     iend,
                  uint64_t                     jstart,
                  uint64_t                     jend,
                  int64_t                      collstart,
                  int64_t                      collend,
                  Flo64                        R0,
                  bool                         collision_interspecies,
                  bool                         SOL,
                  bool                         LIM,
                  const ViewUHL2<const Flo64>& a_host_visible_comb_mat,
                  const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                  const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                  const ViewUHL1<const Flo64>& a_host_visible_mug,
                  const ViewUHL1<const Flo64>& a_host_visible_vparg,
                  const ViewUHL1<const Flo64>& a_host_visible_coeff_intdmu,
                  const ViewUHL1<const Flo64>& a_host_visible_coeff_intdvpar);

            void PrepareNuStar(const Flo64* rg,
                               const Flo64* nustaR0_r,
                               const Flo64* q_r,
                               const Flo64* mask_buffer_r,
                               Flo64*       nustar_coeff) const;

            void ComputeMoments(const Flo64* B_norm,
                                const Flo64* Bstar_s,
                                Flo64*       f5d_s,
                                Flo64*       n_s,
                                Flo64*       U_par_s,
                                Flo64*       T_s) const;

            void ComputeFrequencyNUAB(Flo64    nustar_coeff,
                                      uint64_t speciesA,
                                      uint64_t speciesB,
                                      Flo64    Ta,
                                      Flo64    nb,
                                      Flo64    Tb,
                                      Flo64*   nu_ab) const;

            void ComputeEvolutionMaxwellian(const Flo64* nustar_coeff,
                                            const Flo64* B_norm,
                                            Flo64*       f5d_s,
                                            const Flo64* n_s,
                                            const Flo64* U_par_s,
                                            Flo64*       T_s,
                                            Flo64        dt) const;

            void ComputeQparOnNT(const Flo64* B_norm,
                                 const Flo64* Bstar_s,
                                 const Flo64* f5d_s,
                                 const Flo64* n_s,
                                 const Flo64* U_par_s,
                                 const Flo64* T_s,
                                 Flo64*       q_par_on_nT_s) const;

            void ComputeCpar(const Flo64* rg,
                             const Flo64* B_norm,
                             const Flo64* q_r,
                             const Flo64* nustaR0_r,
                             Flo64*       f5d_s,
                             const Flo64* n_s,
                             const Flo64* U_par_s,
                             const Flo64* T_s,
                             const Flo64* q_par_on_nT_s,
                             Flo64        dt) const;

            void ComputeOperationMaxwellianUnshiftedLoc(const Flo64* B_norm,
                                                        Flo64*       f5d_s,
                                                        const Flo64* n_s,
                                                        const Flo64* T_s,
                                                        uint32_t     choice) const;

        protected:
            void DoPrepareNuStar(const ViewUHL1<const Flo64>& a_host_visible_rg,
                                 const ViewUHL1<const Flo64>& a_host_visible_nustaR0_r,
                                 const ViewUHL1<const Flo64>& a_host_visible_q_r,
                                 const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r,
                                 ViewUHL3<Flo64>&&            a_host_visible_nustar_coeff) const;

            void DoComputeMoments(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                  const ViewUHL4<const Flo64>& a_host_visible_Bstar_s,
                                  ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                  ViewUHL4<Flo64>&&            a_host_visible_n_s,
                                  ViewUHL4<Flo64>&&            a_host_visible_U_par_s,
                                  ViewUHL4<Flo64>&&            a_host_visible_T_s) const;

            void DoComputeEvolutionMaxwellian(const ViewUHL3<const Flo64>& a_host_visible_nustar_coeff,
                                              const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                              ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                              const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                              const ViewUHL4<const Flo64>& a_host_visible_U_par_s,
                                              ViewUHL4<Flo64>&&            a_host_visible_T_s,
                                              Flo64                        dt) const;

            void DoComputeQparOnNT(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                   const ViewUHL4<const Flo64>& a_host_visible_Bstar_s,
                                   const ViewUHL6<const Flo64>& a_host_visible_f5d_s,
                                   const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                   const ViewUHL4<const Flo64>& a_host_visible_U_par_s,
                                   const ViewUHL4<const Flo64>& a_host_visible_T_s,
                                   ViewUHL4<Flo64>&&            a_host_visible_q_par_on_nT_s) const;

            void DoComputeCpar(const ViewUHL1<const Flo64>& a_host_visible_rg,
                               const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                               const ViewUHL1<const Flo64>& a_host_visible_q_r,
                               const ViewUHL1<const Flo64>& a_host_visible_nustaR0_r,
                               ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                               const ViewUHL4<const Flo64>& a_host_visible_n_s,
                               const ViewUHL4<const Flo64>& a_host_visible_U_par_s,
                               const ViewUHL4<const Flo64>& a_host_visible_T_s,
                               const ViewUHL4<const Flo64>& a_host_visible_q_par_on_nT_s,
                               Flo64                        dt) const;

            void DoComputeOperationMaxwellianUnshiftedLoc(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                                          ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                                          const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                                          const ViewUHL4<const Flo64>& a_host_visible_T_s,
                                                          uint32_t                     choice) const;

            static void PrepareCj_Cjl(ViewMDD2<Flo64>& C_j,
                                      ViewMDD3<Flo64>& C_jl);

            static void PrepareCjl(const ViewMDD2<Flo64>::host_mirror_type& a_host_visible_C_j,
                                   ViewMDD3<Flo64>&                         C_jl);

            static KOKKOS_INLINE_FUNCTION Flo64
            DoComputeFrequencyNUAB(Flo64 nustar_coeff,
                                   Flo64 hat_As_speciesA,
                                   Flo64 hat_As_speciesB,
                                   Flo64 Ta,
                                   Flo64 nb,
                                   Flo64 Tb);

            // TODO(Etienne M): Convert all these indicies to half open ranges
            // counter.
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

            ViewMDD2<Flo64> comb_mat_;
            ViewMDD1<Flo64> hat_As_;
            ViewMDD1<Flo64> hat_Zs_;
            ViewMDD1<Flo64> mug_;
            ViewMDD1<Flo64> vparg_;
            ViewMDD1<Flo64> coeff_intdmu_;
            ViewMDD1<Flo64> coeff_intdvpar_;

            ViewMDD2<Flo64> C_j_;
            ViewMDD3<Flo64> C_jl_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

        inline State::State(uint64_t                     Npolmax,
                            uint64_t                     index_max,
                            uint64_t                     Nmu,
                            uint64_t                     Nvpar,
                            uint64_t                     Nspecies,
                            uint64_t                     pSpecies_id,
                            uint64_t                     mu_id,
                            uint64_t                     istart,
                            uint64_t                     iend,
                            uint64_t                     jstart,
                            uint64_t                     jend,
                            int64_t                      collstart,
                            int64_t                      collend,
                            Flo64                        R0,
                            bool                         collision_interspecies,
                            bool                         SOL,
                            bool                         LIM,
                            const ViewUHL2<const Flo64>& a_host_visible_comb_mat,
                            const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                            const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                            const ViewUHL1<const Flo64>& a_host_visible_mug,
                            const ViewUHL1<const Flo64>& a_host_visible_vparg,
                            const ViewUHL1<const Flo64>& a_host_visible_coeff_intdmu,
                            const ViewUHL1<const Flo64>& a_host_visible_coeff_intdvpar)
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

            // TODO(Etienne M): Make sure, we get ALL the data and not only some
            // thread specific part. The collision operator handles its
            // *threads* autonomously.

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

        inline void State::PrepareNuStar(const Flo64* rg,
                                         const Flo64* nustaR0_r,
                                         const Flo64* q_r,
                                         const Flo64* mask_buffer_r,
                                         Flo64*       nustar_coeff) const {

            const uint64_t the_rank_r_range = (iend_ + 1) - istart_;

            // NOTE: For nustar_coeff, we assume the whole array is
            // contiguous and that we can discard the istart_ offset. So
            // assuming:
            // real(F64), contiguous, intent(out) :: nustar_coeff(0:,0:,istart_:)
            // We get:
            // real(F64), contiguous, intent(out) :: nustar_coeff(0:,0:,0:the_rank_r_range - 1)

            DoPrepareNuStar(ViewUHL1<const Flo64>{// "unmanaged_host_rg",
                                                  rg, the_rank_r_range},
                            ViewUHL1<const Flo64>{// "unmanaged_host_nustaR0_r",
                                                  nustaR0_r, the_rank_r_range},
                            ViewUHL1<const Flo64>{// "unmanaged_host_q_r",
                                                  q_r, the_rank_r_range},
                            ViewUHL1<const Flo64>{// "unmanaged_host_mask_buffer_r",
                                                  mask_buffer_r, the_rank_r_range},
                            ViewUHL3<Flo64>{// "unmanaged_host_nustar_coeff",
                                            nustar_coeff, Nspecies_, Nspecies_, the_rank_r_range});
        }

        inline void State::ComputeMoments(const Flo64* B_norm,
                                          const Flo64* Bstar_s,
                                          Flo64*       f5d_s,
                                          Flo64*       n_s,
                                          Flo64*       U_par_s,
                                          Flo64*       T_s) const {

            const uint64_t the_rank_r_range     = (iend_ + 1) - istart_;
            const uint64_t the_rank_theta_range = (jend_ + 1) - jstart_;
            const uint64_t the_rank_phi_range   = static_cast<uint64_t>((collend_ + 1) - collstart_);

            DoComputeMoments(ViewUHL2<const Flo64>{// "unmanaged_host_B_norm",
                                                   B_norm, the_rank_r_range, the_rank_theta_range + /* This +1 is weird */ 1},
                             ViewUHL4<const Flo64>{// "unmanaged_host_Bstar_s",
                                                   Bstar_s, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, Nspecies_},
                             ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                             f5d_s, Nmu_ + 1, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                             ViewUHL4<Flo64>{// "unmanaged_host_n_s",
                                             n_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                             ViewUHL4<Flo64>{// "unmanaged_host_U_par_s",
                                             U_par_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                             ViewUHL4<Flo64>{// "unmanaged_host_T_s",
                                             T_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_});
        }

        inline void State::ComputeFrequencyNUAB(Flo64    nustar_coeff,
                                                uint64_t speciesA,
                                                uint64_t speciesB,
                                                Flo64    Ta,
                                                Flo64    nb,
                                                Flo64    Tb,
                                                Flo64*   nu_ab) const {

            const auto a_host_visible_hat_As = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_As_);

            *nu_ab = DoComputeFrequencyNUAB(nustar_coeff,
                                            a_host_visible_hat_As(speciesA),
                                            a_host_visible_hat_As(speciesB),
                                            Ta,
                                            nb,
                                            Tb);
        }

        inline void State::ComputeEvolutionMaxwellian(const Flo64* nustar_coeff,
                                                      const Flo64* B_norm,
                                                      Flo64*       f5d_s,
                                                      const Flo64* n_s,
                                                      const Flo64* U_par_s,
                                                      Flo64*       T_s,
                                                      Flo64        dt) const {

            const uint64_t the_rank_r_range     = (iend_ + 1) - istart_;
            const uint64_t the_rank_theta_range = (jend_ + 1) - jstart_;
            const uint64_t the_rank_phi_range   = static_cast<uint64_t>((collend_ + 1) - collstart_);

            DoComputeEvolutionMaxwellian(ViewUHL3<const Flo64>{// "unmanaged_host_nustar_coeff",
                                                               nustar_coeff, Nspecies_, Nspecies_, the_rank_r_range},
                                         ViewUHL2<const Flo64>{// "unmanaged_host_B_norm",
                                                               B_norm, the_rank_r_range, the_rank_theta_range + /* This +1 is weird */ 1},
                                         ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                         f5d_s, Nmu_ + 1, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                                         ViewUHL4<const Flo64>{// "unmanaged_host_n_s",
                                                               n_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                                         ViewUHL4<const Flo64>{// "unmanaged_host_U_par_s",
                                                               U_par_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                                         ViewUHL4<Flo64>{// "unmanaged_host_T_s",
                                                         T_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                                         dt);
        }

        inline void State::ComputeQparOnNT(const Flo64* B_norm,
                                           const Flo64* Bstar_s,
                                           const Flo64* f5d_s,
                                           const Flo64* n_s,
                                           const Flo64* U_par_s,
                                           const Flo64* T_s,
                                           Flo64*       q_par_on_nT_s) const {
            const uint64_t the_rank_r_range     = (iend_ + 1) - istart_;
            const uint64_t the_rank_theta_range = (jend_ + 1) - jstart_;
            const uint64_t the_rank_phi_range   = static_cast<uint64_t>((collend_ + 1) - collstart_);

            DoComputeQparOnNT(ViewUHL2<const Flo64>{// "unmanaged_host_B_norm",
                                                    B_norm, the_rank_r_range, the_rank_theta_range + /* This +1 is weird */ 1},
                              ViewUHL4<const Flo64>{// "unmanaged_host_Bstar_s",
                                                    Bstar_s, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, Nspecies_},
                              ViewUHL6<const Flo64>{// "unmanaged_host_f5d_s",
                                                    f5d_s, Nmu_ + 1, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                              ViewUHL4<const Flo64>{// "unmanaged_host_n_s",
                                                    n_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                              ViewUHL4<const Flo64>{// "unmanaged_host_U_par_s",
                                                    U_par_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                              ViewUHL4<const Flo64>{// "unmanaged_host_T_s",
                                                    T_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                              ViewUHL4<Flo64>{// "unmanaged_host_q_par_on_nT_s",
                                              q_par_on_nT_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_});
        }

        inline void State::ComputeCpar(const Flo64* rg,
                                       const Flo64* B_norm,
                                       const Flo64* q_r,
                                       const Flo64* nustaR0_r,
                                       Flo64*       f5d_s,
                                       const Flo64* n_s,
                                       const Flo64* U_par_s,
                                       const Flo64* T_s,
                                       const Flo64* q_par_on_nT_s,
                                       Flo64        dt) const {

            const uint64_t the_rank_r_range     = (iend_ + 1) - istart_;
            const uint64_t the_rank_theta_range = (jend_ + 1) - jstart_;
            const uint64_t the_rank_phi_range   = static_cast<uint64_t>((collend_ + 1) - collstart_);

            DoComputeCpar(ViewUHL1<const Flo64>{// "unmanaged_host_rg",
                                                rg, the_rank_r_range},
                          ViewUHL2<const Flo64>{// "unmanaged_host_B_norm",
                                                B_norm, the_rank_r_range, the_rank_theta_range + /* This +1 is weird */ 1},
                          ViewUHL1<const Flo64>{// "unmanaged_host_q_r",
                                                q_r, the_rank_r_range},
                          ViewUHL1<const Flo64>{// "unmanaged_host_nustaR0_r",
                                                nustaR0_r, the_rank_r_range},
                          ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                          f5d_s, Nmu_ + 1, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                          ViewUHL4<const Flo64>{// "unmanaged_host_n_s",
                                                n_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                          ViewUHL4<const Flo64>{// "unmanaged_host_U_par_s",
                                                U_par_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                          ViewUHL4<const Flo64>{// "unmanaged_host_T_s",
                                                T_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                          ViewUHL4<const Flo64>{// "unmanaged_host_q_par_on_nT_s",
                                                q_par_on_nT_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                          dt);
        }

        inline void State::ComputeOperationMaxwellianUnshiftedLoc(const Flo64* B_norm,
                                                                  Flo64*       f5d_s,
                                                                  const Flo64* n_s,
                                                                  const Flo64* T_s,
                                                                  uint32_t     choice) const {

            const uint64_t the_rank_r_range     = (iend_ + 1) - istart_;
            const uint64_t the_rank_theta_range = (jend_ + 1) - jstart_;
            const uint64_t the_rank_phi_range   = static_cast<uint64_t>((collend_ + 1) - collstart_);

            DoComputeOperationMaxwellianUnshiftedLoc(ViewUHL2<const Flo64>{// "unmanaged_host_B_norm",
                                                                           B_norm, the_rank_r_range, the_rank_theta_range + /* This +1 is weird */ 1},
                                                     ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                                     f5d_s, Nmu_ + 1, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                                                     ViewUHL4<const Flo64>{// "unmanaged_host_n_s",
                                                                           n_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},

                                                     ViewUHL4<const Flo64>{// "unmanaged_host_T_s",
                                                                           T_s, the_rank_r_range, the_rank_theta_range, the_rank_phi_range, Nspecies_},
                                                     choice);
        }

        inline void State::DoPrepareNuStar(const ViewUHL1<const Flo64>& a_host_visible_rg,
                                           const ViewUHL1<const Flo64>& a_host_visible_nustaR0_r,
                                           const ViewUHL1<const Flo64>& a_host_visible_q_r,
                                           const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r,
                                           ViewUHL3<Flo64>&&            a_host_visible_nustar_coeff) const {

            const uint64_t element_count = a_host_visible_rg.extent(0);

            Kokkos::View<Flo64*, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space>
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
            const auto a_host_visible_hat_As = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_As_);
            const auto a_host_visible_hat_Zs = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_Zs_);

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
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
                        // which is independent on the species considered (eps = r/R0_)
                        const Flo64 eps = a_host_visible_rg(ir) * R0_inverse_;

                        const Flo64 AD = a_host_visible_nustaR0_r(ir) *
                                         mask_space(ir) *
                                         // TODO(Etienne M): SIMPLIFY
                                         // sqrt(2)*(rg*R0_inverse_)*sqrt(rg*R0_inverse_)*R0_inverse_ =
                                         // sqrt(2)*pow(r, 5/2)*pow(rg, 3/2)
                                         // sqrt(2)*pow(r, 5/2) is constant here.
                                         koliop::portability::MathematicalConstant<Flo64>::kSQRT2 *
                                         eps *
                                         Kokkos::sqrt(eps) *
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

        inline void State::DoComputeMoments(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                            const ViewUHL4<const Flo64>& a_host_visible_Bstar_s,
                                            ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                            ViewUHL4<Flo64>&&            a_host_visible_n_s,
                                            ViewUHL4<Flo64>&&            a_host_visible_U_par_s,
                                            ViewUHL4<Flo64>&&            a_host_visible_T_s) const {

            const auto a_host_visible_mug            = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           mug_);
            const auto a_host_visible_vparg          = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           vparg_);
            const auto a_host_visible_coeff_intdmu   = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           coeff_intdmu_);
            const auto a_host_visible_coeff_intdvpar = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           coeff_intdvpar_);

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi, auto ispecies) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                Flo64 na    = 0.0;
                Flo64 vpara = 0.0;

                for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        const Flo64 f5d_s_value = std::max(1.0E-40,
                                                           a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const Flo64 ftmp = (a_host_visible_Bstar_s(ivpar, ir, itheta, ispecies) * a_host_visible_coeff_intdvpar(ivpar)) *
                                           (a_host_visible_coeff_intdmu(imu) * f5d_s_value);
                        na += ftmp;
                        vpara += a_host_visible_vparg(ivpar) * ftmp;

                        a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) = f5d_s_value;
                    }
                }

                vpara = vpara / na;

                Flo64 Ta = 0.0;

                for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        // TODO(Etienne M): Avoid this recomputation ? On
                        // Gysela fortran, a temporary buffer is used but
                        // that would hardly fit on a GPU.
                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const Flo64 ftmp = (a_host_visible_Bstar_s(ivpar, ir, itheta, ispecies) * a_host_visible_coeff_intdvpar(ivpar)) *
                                           (a_host_visible_coeff_intdmu(imu) * a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));
                        Ta += ((a_host_visible_vparg(ivpar) - vpara) * (a_host_visible_vparg(ivpar) - vpara) +
                               2 * a_host_visible_mug(imu) * a_host_visible_B_norm(ir, itheta)) *
                              ftmp;
                    }
                }

                Ta = Ta / (3.0 * na);
                na = koliop::portability::MathematicalConstant<Flo64>::k2Pi * na;

                a_host_visible_n_s(ir, itheta, iphi, ispecies)     = na;
                a_host_visible_U_par_s(ir, itheta, iphi, ispecies) = vpara;
                a_host_visible_T_s(ir, itheta, iphi, ispecies)     = Ta;
            };

            const uint64_t the_rank_r_range     = a_host_visible_n_s.extent(0);
            const uint64_t the_rank_theta_range = a_host_visible_n_s.extent(1);
            const uint64_t the_rank_phi_range   = a_host_visible_n_s.extent(2);

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0, 0},
                                                                                           {the_rank_r_range,
                                                                                            the_rank_theta_range,
                                                                                            the_rank_phi_range,
                                                                                            Nspecies_}};

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, the_kernel);
        }

        inline void State::DoComputeEvolutionMaxwellian(const ViewUHL3<const Flo64>& a_host_visible_nustar_coeff,
                                                        const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                                        ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                                        const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                                        const ViewUHL4<const Flo64>& a_host_visible_U_par_s,
                                                        ViewUHL4<Flo64>&&            a_host_visible_T_s,
                                                        Flo64                        dt) const {

            const auto a_host_visible_hat_As = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_As_);
            const auto a_host_visible_mug    = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   mug_);
            const auto a_host_visible_vparg  = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   vparg_);

            // TODO(Etienne M): I wish we had MDRange with scratch memory.
            // Instead of having to use a TeamPolicy and computing MDRange index
            // by hand.

            // NOTE: We use scratch memory to avoid overwriting the T_s.
            using ScratchMemory = Kokkos::View<Flo64*, Kokkos::DefaultHostExecutionSpace::scratch_memory_space>;

            const uint64_t the_scratch_memory_bytes_per_thread = Nspecies_ * sizeof(Flo64);

            using KernelIterationPolicy = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;

            const uint64_t the_rank_r_range     = a_host_visible_n_s.extent(0);
            const uint64_t the_rank_theta_range = a_host_visible_n_s.extent(1);
            const uint64_t the_rank_phi_range   = a_host_visible_n_s.extent(2);

            const uint64_t the_team_count = the_rank_theta_range * the_rank_phi_range;
            // Due to the inherent cuda/HIP threadblock limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {
                // Reference to the_kernel's member variable copy.
                const auto the_kernel_core = [&](auto ir, auto itheta, auto iphi, auto& Ta_tmp) {
                    // NOTE: Iterating on species leads to large strides in
                    // memory access. This is mitigated on GPU but painful on
                    // CPU. This behavior done to avoid copying the whole Ts_.
                    // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                    for(uint64_t ispeciesA = 0; ispeciesA < Nspecies_; ++ispeciesA) {
                        const Flo64 hat_Aa = a_host_visible_hat_As(ispeciesA);
                        const Flo64 na     = a_host_visible_n_s(ir, itheta, iphi, ispeciesA);
                        const Flo64 Vpara  = a_host_visible_U_par_s(ir, itheta, iphi, ispeciesA);
                        const Flo64 Ta     = a_host_visible_T_s(ir, itheta, iphi, ispeciesA);

                        Flo64 Ta_new = Ta;

                        // TODO(Etienne M): Split into two loops to avoid the if ?

                        for(uint64_t ispeciesB = 0; ispeciesB < Nspecies_; ++ispeciesB) {
                            if(ispeciesA == ispeciesB) {
                                continue;
                            }

                            const Flo64 hat_Ab = a_host_visible_hat_As(ispeciesB);
                            const Flo64 nb     = a_host_visible_n_s(ir, itheta, iphi, ispeciesB);
                            const Flo64 Vparb  = a_host_visible_U_par_s(ir, itheta, iphi, ispeciesB);
                            const Flo64 Tb     = a_host_visible_T_s(ir, itheta, iphi, ispeciesB);

                            const Flo64 nu_ab = DoComputeFrequencyNUAB(a_host_visible_nustar_coeff(ispeciesA, ispeciesB, ir),
                                                                       a_host_visible_hat_As(ispeciesA),
                                                                       a_host_visible_hat_As(ispeciesB),
                                                                       Ta, nb, Tb);

                            Ta_new += dt * nu_ab * (2.0 * hat_Aa * (Tb - Ta) / (hat_Aa + hat_Ab) + (2.0 / 3.0) * Vpara * (Vpara - sqrt(hat_Aa / hat_Ab) * Vparb));
                        }

                        // We can not overwrite the T_s until all the new_T_s are
                        // computed.
                        Ta_tmp(ispeciesA) = Ta_new;

                        const Flo64 coeff_old = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta *
                                                      Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));
                        const Flo64 coeff_new = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta_new *
                                                      Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta_new));
                        const Flo64 Bij       = a_host_visible_B_norm(ir, itheta);
                        const Flo64 bot_old   = Bij / Ta;
                        const Flo64 bot_new   = Bij / Ta_new;

                        for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                            const Flo64 vp2_old = (a_host_visible_vparg(ivpar) - Vpara) * (a_host_visible_vparg(ivpar) - Vpara) / (2.0 * Ta);
                            const Flo64 vp2_new = (a_host_visible_vparg(ivpar) - Vpara) * (a_host_visible_vparg(ivpar) - Vpara) / (2.0 * Ta_new);
                            for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                                a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispeciesA) -= coeff_old * Kokkos::exp(-bot_old * a_host_visible_mug(imu) - vp2_old) +
                                                                                                 coeff_new * Kokkos::exp(-bot_new * a_host_visible_mug(imu) - vp2_new);
                            }
                        }
                    }

                    for(uint64_t ispeciesA = 0; ispeciesA < Nspecies_; ++ispeciesA) {
                        a_host_visible_T_s(ir, itheta, iphi, ispeciesA) = Ta_tmp(ispeciesA);
                    }
                };

                const auto the_team_index = a_team_handle.league_rank();

                const auto iphi   = the_team_index / the_rank_theta_range;
                const auto itheta = the_team_index % the_rank_theta_range;

                // TODO(Etienne M): Should be put that view allocation in the
                // parallel_for below ? I dont think so as it would lead to
                // bumping the LDS pointer for each R iteration. I think this
                // should be done once per thread.
                // TODO(Etienne M): Should we a_team_handle.team_barrier() ?
                ScratchMemory the_thread_scratch_memory{a_team_handle.team_scratch(0),
                                                        the_scratch_memory_bytes_per_thread};

                const auto a_policy = Kokkos::TeamThreadRange(a_team_handle,
                                                              the_rank_r_range);

                Kokkos::parallel_for(a_policy,
                                     // Reference to the_kernel's member variable copy.
                                     [&](auto ir) {
                                         the_kernel_core(ir, itheta, iphi, the_thread_scratch_memory);
                                     });
            };

            auto a_policy = KernelIterationPolicy{static_cast<int>(the_team_count),
                                                  Kokkos::AUTO};

            a_policy.set_scratch_size(0,
                                      Kokkos::PerThread(ScratchMemory::shmem_size(the_scratch_memory_bytes_per_thread)));

            Kokkos::parallel_for("DoComputeEvolutionMaxwellian",
                                 a_policy, the_kernel);
        }

        inline void State::DoComputeQparOnNT(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                             const ViewUHL4<const Flo64>& a_host_visible_Bstar_s,
                                             const ViewUHL6<const Flo64>& a_host_visible_f5d_s,
                                             const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                             const ViewUHL4<const Flo64>& a_host_visible_U_par_s,
                                             const ViewUHL4<const Flo64>& a_host_visible_T_s,
                                             ViewUHL4<Flo64>&&            a_host_visible_q_par_on_nT_s) const {

            const auto a_host_visible_mug            = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           mug_);
            const auto a_host_visible_vparg          = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           vparg_);
            const auto a_host_visible_coeff_intdmu   = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           coeff_intdmu_);
            const auto a_host_visible_coeff_intdvpar = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                              Kokkos::WithoutInitializing),
                                                                                           coeff_intdvpar_);

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi, auto ispecies) {
                // TODO(Etienne M): Should we flatten even more the loops ?
                // Potentially computing/loading redundant values (Bij, na etc.)
                // but extracting more parallelisms. The thing is that ivpar and
                // imu tends to be larger than ir, itheta or iphi.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 Bij   = a_host_visible_B_norm(ir, itheta);
                const Flo64 na    = a_host_visible_n_s(ir, itheta, iphi, ispecies);
                const Flo64 Vpara = a_host_visible_U_par_s(ir, itheta, iphi, ispecies);
                const Flo64 Ta    = a_host_visible_T_s(ir, itheta, iphi, ispecies);
                // TODO(Etienne M): SIMPLIFY as std::pow(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta, -3 / 2);
                const Flo64 coeff = 1.0 / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));

                Flo64 q_par_on_nT_s_reduction = 0.0;

                for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                    const Flo64 jac = koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                      a_host_visible_Bstar_s(ivpar, ir, itheta, ispecies);

                    const Flo64 vparg_minus_vpara         = a_host_visible_vparg(ivpar) - Vpara;
                    const Flo64 vparg_minus_vpara_squared = vparg_minus_vpara * vparg_minus_vpara;
                    const Flo64 vp2                       = vparg_minus_vpara_squared / (2.0 * Ta);

                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        const Flo64 u_tmp   = a_host_visible_mug(imu) * Bij / Ta;
                        const Flo64 FMa_tmp = na * Kokkos::exp(-u_tmp - vp2) * coeff;
                        const Flo64 v2      = 2.0 * a_host_visible_mug(imu) * Bij + vparg_minus_vpara_squared;
                        q_par_on_nT_s_reduction += 0.5 *
                                                   jac *
                                                   a_host_visible_coeff_intdmu(imu) *
                                                   a_host_visible_coeff_intdvpar(ivpar) *
                                                   vparg_minus_vpara *
                                                   v2 *
                                                   (a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) - FMa_tmp);
                    }
                }

                a_host_visible_q_par_on_nT_s(ir, itheta, iphi, ispecies) = q_par_on_nT_s_reduction / (na * Ta);
            };

            const uint64_t the_rank_r_range     = a_host_visible_n_s.extent(0);
            const uint64_t the_rank_theta_range = a_host_visible_n_s.extent(1);
            const uint64_t the_rank_phi_range   = a_host_visible_n_s.extent(2);

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0, 0},
                                                                                           {the_rank_r_range,
                                                                                            the_rank_theta_range,
                                                                                            the_rank_phi_range,
                                                                                            Nspecies_}};

            Kokkos::parallel_for("DoComputeQparOnNT",
                                 a_policy, the_kernel);
        }

        inline void State::DoComputeCpar(const ViewUHL1<const Flo64>& a_host_visible_rg,
                                         const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                         const ViewUHL1<const Flo64>& a_host_visible_q_r,
                                         const ViewUHL1<const Flo64>& a_host_visible_nustaR0_r,
                                         ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                         const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                         const ViewUHL4<const Flo64>& a_host_visible_U_par_s,
                                         const ViewUHL4<const Flo64>& a_host_visible_T_s,
                                         const ViewUHL4<const Flo64>& a_host_visible_q_par_on_nT_s,
                                         Flo64                        dt) const {

            const auto a_host_visible_hat_As = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_As_);
            const auto a_host_visible_hat_Zs = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_Zs_);

            const auto a_host_visible_mug   = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                     Kokkos::WithoutInitializing),
                                                                                  mug_);
            const auto a_host_visible_vparg = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                     Kokkos::WithoutInitializing),
                                                                                  vparg_);

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): Assuming materialization, we could flatten
                // the loop even more, up to Nmu_.
                // TODO(Etienne M): We could inter exchange some loops ?
                // TODO(Etienne M): There is quite a lot to simplify here.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 hat_Aa     = a_host_visible_hat_As(pSpecies_id_);
                const Flo64 hat_Za     = a_host_visible_hat_Zs(pSpecies_id_);
                const Flo64 inv_hat_Aa = 1.0 / hat_Aa;

                const Flo64 smcsfqri = a_host_visible_q_r(ir);
                const Flo64 Bij      = a_host_visible_B_norm(ir, itheta);

                const Flo64 na          = a_host_visible_n_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 Vpara       = a_host_visible_U_par_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 Ta          = a_host_visible_T_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 q_par_on_nT = a_host_visible_q_par_on_nT_s(ir, itheta, iphi, pSpecies_id_);

                const Flo64 nustaR0_i = a_host_visible_nustaR0_r(ir);
                const Flo64 epstemp   = a_host_visible_rg(ir) * R0_inverse_;
                const Flo64 vTa       = Kokkos::sqrt(Ta / hat_Aa);
                const Flo64 coeff1    = 1.0 / (2.0 * Ta);
                const Flo64 coeff3    = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));
                const Flo64 AD        = nustaR0_i * koliop::portability::MathematicalConstant<Flo64>::kSQRT2 * epstemp * Kokkos::sqrt(epstemp) / ((1.0 / R0_inverse_) * (smcsfqri != koliop::portability::GyselaConstant<Flo64>::kInfinity ? smcsfqri : 1.0));

                for(uint64_t ispecies = 0; ispecies < Nspecies_; ++ispecies) {
                    const Flo64 hat_Ab         = a_host_visible_hat_As(ispecies);
                    const Flo64 inv_hat_Ab     = 1.0 / hat_Ab;
                    const Flo64 hat_Zb         = a_host_visible_hat_Zs(ispecies);
                    const Flo64 nb             = a_host_visible_n_s(ir, itheta, iphi, ispecies);
                    const Flo64 Vparb          = a_host_visible_U_par_s(ir, itheta, iphi, ispecies);
                    const Flo64 Tb             = a_host_visible_T_s(ir, itheta, iphi, ispecies);
                    const Flo64 q_parb_on_nbTb = a_host_visible_q_par_on_nT_s(ir, itheta, iphi, ispecies);

                    const Flo64 Rat    = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const Flo64 vTb    = Kokkos::sqrt(Tb / hat_Ab);
                    const Flo64 coeff2 = vTa / vTb;
                    const Flo64 coeff4 = (3.0 / 5.0) * Rat * q_parb_on_nbTb / (1.0 + (Ta * hat_Ab) / (Tb * hat_Aa));

                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                            const Flo64 coeff5         = (a_host_visible_vparg(ivpar) / Ta);
                            const Flo64 xa             = Kokkos::sqrt(coeff1 * (2.0 * a_host_visible_mug(imu) * Bij + a_host_visible_vparg(ivpar) * a_host_visible_vparg(ivpar)));
                            const Flo64 xb             = xa * coeff2;
                            const Flo64 FM0a           = coeff3 * Kokkos::exp(-xa * xa);
                            const Flo64 Theta_func_tmp = xb > 1E-15 ?
                                                             koliop::portability::GyselaConstant<Flo64>::kConst_theta * ((Kokkos::erf(xb) - xb * (koliop::portability::GyselaConstant<Flo64>::kCst_phiprime * exp(-xb * xb))) / (2.0 * xb * xb)) / xb :
                                                             1.0;
                            const Flo64 nu_sab         = (AD * hat_Za * hat_Za * hat_Zb * hat_Zb * nb) * inv_hat_Aa * (inv_hat_Aa + inv_hat_Ab) * Theta_func_tmp / (vTb * vTb * vTb);

                            a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_) += dt * FM0a * nu_sab * coeff5 * (Rat * Vparb - Vpara + q_par_on_nT * (1.0 - 2.0 * xa * xa * 0.2) - coeff4);
                        }
                    }
                }
            };

            const uint64_t the_rank_r_range     = a_host_visible_n_s.extent(0);
            const uint64_t the_rank_theta_range = a_host_visible_n_s.extent(1);
            const uint64_t the_rank_phi_range   = a_host_visible_n_s.extent(2);

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0},
                                                                                           {the_rank_r_range,
                                                                                            the_rank_theta_range,
                                                                                            the_rank_phi_range}};

            Kokkos::parallel_for("DoComputeCpar",
                                 a_policy, the_kernel);
        }

        inline void State::DoComputeOperationMaxwellianUnshiftedLoc(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                                                    ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                                                    const ViewUHL4<const Flo64>& a_host_visible_n_s,
                                                                    const ViewUHL4<const Flo64>& a_host_visible_T_s,
                                                                    uint32_t                     choice) const {

            const auto a_host_visible_mug   = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                     Kokkos::WithoutInitializing),
                                                                                  mug_);
            const auto a_host_visible_vparg = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                     Kokkos::WithoutInitializing),
                                                                                  vparg_);

            const auto the_kernel_choice_3 = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                const Flo64 Bij    = a_host_visible_B_norm(ir, itheta);
                const Flo64 na     = a_host_visible_n_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 Ta     = a_host_visible_T_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 BijoTa = Bij / Ta;
                const Flo64 f2oTa  = 0.5 / Ta;
                const Flo64 coeff  = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));

                for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        const Flo64 v2oTa = a_host_visible_mug(imu) * BijoTa + a_host_visible_vparg(ivpar) * a_host_visible_vparg(ivpar) * f2oTa;

                        a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_) = std::max(0.0,
                                                                                                    coeff * exp(-v2oTa) * a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_));
                    }
                }
            };

            const auto the_kernel_choice_4 = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                const Flo64 Bij    = a_host_visible_B_norm(ir, itheta);
                const Flo64 na     = a_host_visible_n_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 Ta     = a_host_visible_T_s(ir, itheta, iphi, pSpecies_id_);
                const Flo64 BijoTa = Bij / Ta;
                const Flo64 f2oTa  = 0.5 / Ta;
                const Flo64 coeff  = (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta)) / na;
                for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        const Flo64 v2oTa = a_host_visible_mug(imu) * BijoTa + a_host_visible_vparg(ivpar) * a_host_visible_vparg(ivpar) * f2oTa;

                        a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_) = std::max(0.001,
                                                                                                    std::min(1000.0,
                                                                                                             coeff * exp(v2oTa) * a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_)));
                    }
                }
            };

            const uint64_t the_rank_r_range     = a_host_visible_n_s.extent(0);
            const uint64_t the_rank_theta_range = a_host_visible_n_s.extent(1);
            const uint64_t the_rank_phi_range   = a_host_visible_n_s.extent(2);

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0},
                                                                                           {the_rank_r_range,
                                                                                            the_rank_theta_range,
                                                                                            the_rank_phi_range}};

            switch(choice) {
                case 3:
                    Kokkos::parallel_for("DoComputeOperationMaxwellianUnshiftedLoc3",
                                         a_policy, the_kernel_choice_3);
                    break;
                case 4:
                    Kokkos::parallel_for("DoComputeOperationMaxwellianUnshiftedLoc4",
                                         a_policy, the_kernel_choice_4);
                    break;
                default:
                    // TODO(Etienne M): Should this case be reachable ?
                    KOLIOP_ASSERT(false);
                    break;
            }
        }

        inline void State::PrepareCj_Cjl(ViewMDD2<Flo64>& C_j,
                                         ViewMDD3<Flo64>& C_jl) {
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

        inline void State::PrepareCjl(const ViewMDD2<Flo64>::host_mirror_type& a_host_visible_C_j,
                                      ViewMDD3<Flo64>&                         C_jl) {
            // a_host_visible_C_jl is zero initialized.
            auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl);

            const auto Stuff = [&](auto j_tmp, auto l_tmp) {
                for(uint64_t i = 0; i <= (j_tmp + l_tmp); ++i) {
                    // This seems unnecessary ?
                    a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                }

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

        KOKKOS_INLINE_FUNCTION Flo64
        State::DoComputeFrequencyNUAB(Flo64 nustar_coeff,
                                      Flo64 hat_As_speciesA,
                                      Flo64 hat_As_speciesB,
                                      Flo64 Ta,
                                      Flo64 nb,
                                      Flo64 Tb) {

            const Flo64 vTa = Kokkos::sqrt(Ta / hat_As_speciesA);
            const Flo64 vTb = Kokkos::sqrt(Tb / hat_As_speciesB);

            // NOTE: SIMPLIFY We could do:
            // (nustar_coeff * nb) *
            //     std::pow((Tb * hat_As_speciesA +
            //               Ta * hat_As_speciesB) /
            //                  (hat_As_speciesA * hat_As_speciesB),
            //              -3.0 / 2.0);
            return (nustar_coeff * nb) / ((vTa * vTa + vTb * vTb) * Kokkos::sqrt(vTa * vTa + vTb * vTb));
        }

    } // namespace collision
} // namespace koliop

#endif
