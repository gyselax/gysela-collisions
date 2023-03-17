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

            DoComputeMoments(ViewUHL2<const Flo64>{// "unmanaged_host_B_norm",
                                                   B_norm, the_rank_r_range, the_rank_theta_range + /* This +1 is weird */ 1},
                             ViewUHL4<const Flo64>{// "unmanaged_host_Bstar_s",
                                                   Bstar_s, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, Nspecies_},
                             ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                             f5d_s, Nmu_ + 1, Nvpar_ + 1, the_rank_r_range, the_rank_theta_range, static_cast<uint64_t>((collend_ + 1) - collstart_), Nspecies_},
                             ViewUHL4<Flo64>{// "unmanaged_host_n_s",
                                             n_s, the_rank_r_range, the_rank_theta_range, static_cast<uint64_t>((collend_ + 1) - collstart_), Nspecies_},
                             ViewUHL4<Flo64>{// "unmanaged_host_U_par_s",
                                             U_par_s, the_rank_r_range, the_rank_theta_range, static_cast<uint64_t>((collend_ + 1) - collstart_), Nspecies_},
                             ViewUHL4<Flo64>{// "unmanaged_host_T_s",
                                             T_s, the_rank_r_range, the_rank_theta_range, static_cast<uint64_t>((collend_ + 1) - collstart_), Nspecies_});
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

            // TODO(Etienne M): Obvious loop inter exchange optimization ?
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

        inline void State::DoComputeMoments(const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                                            const ViewUHL4<const Flo64>& a_host_visible_Bstar_s,
                                            ViewUHL6<Flo64>&&            a_host_visible_f5d_s,
                                            ViewUHL4<Flo64>&&            a_host_visible_n_s,
                                            ViewUHL4<Flo64>&&            a_host_visible_U_par_s,
                                            ViewUHL4<Flo64>&&            a_host_visible_T_s) const {

            const uint64_t the_rank_r_range = a_host_visible_B_norm.extent(0);

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

            const auto the_kernel = KOKKOS_LAMBDA(auto itheta, auto iphi, auto ispecies) {
                for(uint64_t ir = 0; ir < the_rank_r_range; ++ir) {
                    auto na    = 0.0;
                    auto vpara = 0.0;

                    for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                        for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                            const auto f5d_s_value = std::max(1.0E-40,
                                                              a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                            // TODO(Etienne M): Hoist the invariant load.
                            const auto ftmp = (a_host_visible_Bstar_s(ivpar, ir, itheta, ispecies) * a_host_visible_coeff_intdvpar(ivpar)) *
                                              (a_host_visible_coeff_intdmu(imu) * f5d_s_value);
                            na += ftmp;
                            vpara += a_host_visible_vparg(ivpar) * ftmp;

                            a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) = f5d_s_value;
                        }
                    }

                    vpara = vpara / na;

                    auto Ta = 0.0;

                    for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                        for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                            // TODO(Etienne M): Avoid this recomputation ? On
                            // Gysela fortran, a temporary buffer is used but
                            // that would hardly fit on a GPU.
                            // TODO(Etienne M): Hoist the invariant load.
                            const auto ftmp = (a_host_visible_Bstar_s(ivpar, ir, itheta, ispecies) * a_host_visible_coeff_intdvpar(ivpar)) *
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
                }
            };

            const uint64_t the_rank_theta_range = (jend_ + 1) - jstart_;

            // TODO(Etienne M): Launch a kernel per species on different
            // execution space instances (partition_space) and loop on phi,
            // theta and r.
            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0},
                                                                                           {the_rank_theta_range,
                                                                                            static_cast<uint64_t>((collend_ + 1) - collstart_),
                                                                                            Nspecies_}};

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, the_kernel);
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
        State::DoComputeFrequencyNUAB(Flo64  nustar_coeff,
                                      Flo64  hat_As_speciesA,
                                      Flo64  hat_As_speciesB,
                                      Flo64  Ta,
                                      Flo64  nb,
                                      Flo64  Tb) {

            const auto vTa = std::sqrt(Ta / hat_As_speciesA);
            const auto vTb = std::sqrt(Tb / hat_As_speciesB);

            // NOTE: SIMPLIFY We could do:
            // (nustar_coeff * nb) *
            //     std::pow((Tb * hat_As_speciesA +
            //               Ta * hat_As_speciesB) /
            //                  (hat_As_speciesA * hat_As_speciesB),
            //              -3.0 / 2.0);
            return (nustar_coeff * nb) / ((vTa * vTa + vTb * vTb) * std::sqrt(vTa * vTa + vTb * vTb));
        }

    } // namespace collision
} // namespace koliop

#endif
