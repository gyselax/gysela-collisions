// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/blas/lu.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        // TODO(Etienne M): Cleanup that stuff.
        // View[U|M][H|D][L|R|D][1-9]
        // Unmanaged/Managed | Host or Default (device or host) | Left or Right layout | Dimension
        template <typename Extent>
        using ViewUHL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        template <typename Float>
        using ViewUHL1 = ViewUHL<Float*>;
        template <typename Float>
        using ViewUHL2 = ViewUHL<Float**>;
        template <typename Float>
        using ViewUHL3 = ViewUHL<Float***>;
        template <typename Float>
        using ViewUHL4 = ViewUHL<Float****>;
        template <typename Float>
        using ViewUHL5 = ViewUHL<Float*****>;
        template <typename Float>
        using ViewUHL6 = ViewUHL<Float******>;

        template <typename Extent>
        using ViewMDL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDL1 = ViewMDL<Float*>;
        template <typename Float>
        using ViewMDL2 = ViewMDL<Float**>;
        template <typename Float>
        using ViewMDL3 = ViewMDL<Float***>;
        template <typename Float>
        using ViewMDL4 = ViewMDL<Float****>;
        template <typename Float>
        using ViewMDL5 = ViewMDL<Float*****>;
        template <typename Float>
        using ViewMDL6 = ViewMDL<Float******>;

        template <typename Extent>
        using ViewMDD = Kokkos::View<Extent, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::DefaultExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMDD1 = ViewMDD<Float*>;
        template <typename Float>
        using ViewMDD2 = ViewMDD<Float**>;
        template <typename Float>
        using ViewMDD3 = ViewMDD<Float***>;
        template <typename Float>
        using ViewMDD4 = ViewMDD<Float****>;
        template <typename Float>
        using ViewMDD5 = ViewMDD<Float*****>;
        template <typename Float>
        using ViewMDD6 = ViewMDD<Float******>;

        template <typename Extent>
        using ViewMHL = Kokkos::View<Extent, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space>;
        template <typename Float>
        using ViewMHL1 = ViewMHL<Float*>;


        KOKKOS_INLINE_FUNCTION Flo64
        DoComputeFrequencyNuab(Flo64 nustar_coeff,
                               Flo64 hat_As_speciesA,
                               Flo64 hat_As_speciesB,
                               Flo64 Ta,
                               Flo64 nb,
                               Flo64 Tb) {

            const Flo64 vTa = Ta / hat_As_speciesA;
            const Flo64 vTb = Tb / hat_As_speciesB;

            // TODO(Etienne M): SIMPLIFY We could do:
            // (nustar_coeff * nb) *
            //     std::pow((Tb * hat_As_speciesA +
            //               Ta * hat_As_speciesB) /
            //                  (hat_As_speciesA * hat_As_speciesB),
            //              -3.0 / 2.0);
            return (nustar_coeff * nb) / ((vTa + vTb) * Kokkos::sqrt(vTa + vTb));
        }

        KOKKOS_INLINE_FUNCTION Flo64
        DoComputeLaguerrePol(uint64_t l, Flo64 x) {
            // TODO(Etienne M): find a better way to compute that, based on l.
            switch(l) {
                case 0:
                    return 1.0;
                case 1:
                    return -x + 1.0;
                case 2:
                    return (1.0 / 2.0) + (-4 + /* 1 * */ x) * x;
                case 3:
                    // Horner -> dependency            : 1 + (-3 + (1.5 - 1/6 * x) * x) * x
                    // but FMA friendly.
                    // Naive  -> latency and dependency: (1.0 / 6.0) * (-(x * x * x) + 9.0 * x * x - 18.0 * x + 6.0);
                    // Slow.
                    // Factorized -> low latency       : ((-1.0 / 6.0) * (x - 0.41577455678347908331)) * ((x - 2.29428036027904171982) * (x - 6.28994508293747919686))
                    return 1.0 + (-3.0 + (1.5 - (1.0 / 6.0) * x) * x) * x;
                default:
                    return -1.0;
            }
        }


        ////////////////////////////////////////////////////////////////////////
        // State definition
        ////////////////////////////////////////////////////////////////////////

        /// TODO(Etinne M): INDEX We have a dilemma. CPU allows for large
        /// temporary, per thread, buffers. GPU do not (at least for LDS). Due
        /// to caches hierarchy, CPU like per thread, contiguous memory
        /// accesses. GPU do not because this kind of access leads to wastes in
        /// the memory transactions (notion of un-coalesced accesses). Using
        /// Kokkos, we hope to attain decent performance on both GPU and CPU.
        ///
        /// Assuming the following example, all arrays with layout left
        /// (Fortran).
        ///
        ///     collapse(3)
        ///     for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///         for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///             for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                 const Flo64 a = A(ir, itheta);
        ///                 const Flo64 b = B(ir, itheta, iphi);
        ///                 for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///                     const Flo64 c = C(ivpar);
        ///                     for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                         const Flo64 d = D(imu);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// Collapsing on the first 3 loops would lead to decent performance on
        /// CPU. On GPU though, this would be crappy as each thread would do
        /// linear access in C, D and E. Access in C and D are *ok* that bad,
        /// because, out of one memory transaction, conceptualize it as a cache
        /// line, only 8 bytes will be kept but the next fetch will be likely to
        /// hit the cache. Except that on MI250X cards, we have very few cache
        /// (8 Mio per GCD!). Access in E are bad because each thread gets a
        /// different ir value which means, thread n+1 will access memory in a
        /// stride of mumax*vparmax (strongly un-coalesced).
        ///
        /// We could try to inter-exchange the loops but this would then work
        /// badly on CPU.
        ///
        ///     collapse(3)
        ///     for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///         for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///             for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                 const Flo64 a = A(ir, itheta);
        ///                 const Flo64 c = C(ivpar);
        ///                 const Flo64 d = D(imu);
        ///                 for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///                     for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                         const Flo64 b = B(ir, itheta, iphi);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// In the end, the most versatile is to collapse everything, do
        /// redundant computation on GPU but dont waste what you bring from
        /// global memory and hope that the compiler will optimize the redundant
        /// loads and computations computation on CPU.
        ///
        ///     collapse(5)
        ///     for(uint64_t itheta = 0; itheta <= thetamax; ++itheta) {
        ///         for(uint64_t iphi = 0; iphi <= phimax; ++iphi) {
        ///             for(uint64_t ir = 0; ir <= rmax; ++ir) {
        ///                 for(uint64_t ivpar = 0; ivpar <= vparmax; ++ivpar) {
        ///                     for(uint64_t imu = 0; imu <= mumax; ++imu) {
        ///                         const Flo64 a = A(ir, itheta);
        ///                         const Flo64 c = C(ivpar);
        ///                         const Flo64 d = D(imu);
        ///                         const Flo64 b = B(ir, itheta, iphi);
        ///                         E(imu, ivpar, ir, itheta, iphi) *= a + b + c + d;
        ///                     }
        ///                 }
        ///             }
        ///         }
        ///     }
        ///
        /// TODO(Etienne M): Convert all these indicies to half open ranges
        /// counter.
        ///
        class State {
        public:
            static inline constexpr uint64_t kMaximumLaguerrePolynomialDegree = 3;

        public:
            using AlphaSolverType = koliop::blas::LUBlockTridiagonalByBlock<Flo64,
                                                                            Kokkos::DefaultExecutionSpace,
                                                                            kMaximumLaguerrePolynomialDegree>;

            // TODO(Etienne M): Tune that size or rework batching
            static inline constexpr uint64_t kBlockExtentR     = 16;
            static inline constexpr uint64_t kBlockExtentTheta = 1;
            static inline constexpr uint64_t kBlockExtentPhi   = 1;

            static inline constexpr uint64_t kFlattenedExtent = kBlockExtentR * kBlockExtentTheta * kBlockExtentPhi;

        public:
            State(uint64_t                     pSpecies_id,
                  uint64_t                     mu_id,
                  uint64_t                     the_mu_extent,
                  uint64_t                     the_vpar_extent,
                  uint64_t                     the_r_extent,
                  uint64_t                     the_theta_extent,
                  uint64_t                     the_phi_extent,
                  uint64_t                     the_species_extent,
                  Flo64                        R0,
                  bool                         collision_interspecies,
                  bool                         SOL,
                  bool                         LIM,
                  uint64_t                     ir_SOL_separatrix,
                  Flo64                        dt,
                  const ViewUHL2<const Flo64>& a_host_visible_comb_mat,
                  const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                  const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                  const ViewUHL1<const Flo64>& a_host_visible_mug,
                  const ViewUHL1<const Flo64>& a_host_visible_vparg,
                  const ViewUHL1<const Flo64>& a_host_visible_coeff_intdmu,
                  const ViewUHL1<const Flo64>& a_host_visible_coeff_intdvpar,
                  const ViewUHL1<const Flo64>& a_host_visible_rg,
                  const ViewUHL1<const Flo64>& a_host_visible_nustar0_r,
                  const ViewUHL1<const Flo64>& a_host_visible_q_r,
                  const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r,
                  const ViewUHL2<const Flo64>& a_host_visible_mask_LIM,
                  const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                  const ViewUHL4<const Flo64>& a_host_visible_Bstar_s);

            void ComputeMoments(Flo64* f5d_s);

            void PrecomputeNu();

            void ComputeMaxwellianEvolution(Flo64* f5d_s);

            void ComputeQparOnNT(const Flo64* f5d_s);

            void ComputeCpar(Flo64* f5d_s) const;

            void ComputeMaxwellianUnshiftedLoc(Flo64*   f5d_s,
                                               uint32_t choice) const;

            void Apply(Flo64* f5d_s) const;

            void ComputeDensityCorrection(Flo64* f5d_s) const;

            void ComputeUparTCorrection(Flo64* f5d_s) const;

        public:
            template <typename View3D,
                      typename View2D>
            void DoPrepareCj_Cjl(View2D& C_j,
                                 View3D& C_jl) const;

            template <typename View3D,
                      typename MirrorView2D>
            void DoPrepareCjl(const MirrorView2D& a_host_visible_C_j,
                              View3D&             C_jl) const;

            void DoPrepareNuStar(ViewMDD3<Flo64>&             nustar_coeff,
                                 const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                                 const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                                 const ViewUHL1<const Flo64>& a_host_visible_rg,
                                 const ViewUHL1<const Flo64>& a_host_visible_nustar0_r,
                                 const ViewUHL1<const Flo64>& a_host_visible_q_r,
                                 const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r) const;

            // Constant, externally computed values

            // TODO(Etienne M): Cleanup unused values
            uint64_t pSpecies_id_;
            uint64_t mu_id_;
            uint64_t the_mu_extent_;
            uint64_t the_vpar_extent_;
            uint64_t the_r_extent_;
            uint64_t the_theta_extent_;
            uint64_t the_phi_extent_;
            uint64_t the_species_extent_;
            Flo64    R0_;
            bool     collision_interspecies_;
            bool     SOL_;
            bool     LIM_;
            uint64_t ir_SOL_separatrix_;
            Flo64    dt_;

            ViewMDD2<Flo64> comb_mat_; // TODO(Etienne M): constant extent.
            ViewMDD1<Flo64> hat_As_;   // TODO(Etienne M): constant extent.
            ViewMDD1<Flo64> hat_Zs_;   // TODO(Etienne M): constant extent.
            ViewMDD1<Flo64> mug_;
            ViewMDD1<Flo64> vparg_;
            ViewMDD1<Flo64> coeff_intdmu_;
            ViewMDD1<Flo64> coeff_intdvpar_;
            ViewMDD1<Flo64> rg_;
            ViewMDD1<Flo64> nustar0_r_;
            ViewMDD1<Flo64> q_r_;
            ViewMDD1<Flo64> mask_buffer_r_;
            ViewMDD2<Flo64> mask_LIM_;
            ViewMDD2<Flo64> B_norm_;
            ViewMDD4<Flo64> Bstar_s_;

            // Constant, internally computed values

            ViewMDD2<Flo64> C_j_;  // TODO(Etienne M): constant extent.
            ViewMDD3<Flo64> C_jl_; // TODO(Etienne M): constant extent.
            ViewMDD3<Flo64> nustar_coeff_;

            // Variable, internally computed values

            ViewMDD4<Flo64> n_s_;
            ViewMDD4<Flo64> U_par_s_;
            ViewMDD4<Flo64> T_s_;
            ViewMDD5<Flo64> nu_ab_;
            ViewMDD4<Flo64> q_par_on_nT_s_;

            // Miscellaneous tooling

            AlphaSolverType the_alpha_solver_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

        inline State::State(uint64_t                     pSpecies_id,
                            uint64_t                     mu_id,
                            uint64_t                     the_mu_extent,
                            uint64_t                     the_vpar_extent,
                            uint64_t                     the_r_extent,
                            uint64_t                     the_theta_extent,
                            uint64_t                     the_phi_extent,
                            uint64_t                     the_species_extent,
                            Flo64                        R0,
                            bool                         collision_interspecies,
                            bool                         SOL,
                            bool                         LIM,
                            uint64_t                     ir_SOL_separatrix,
                            Flo64                        dt,
                            const ViewUHL2<const Flo64>& a_host_visible_comb_mat,
                            const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                            const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                            const ViewUHL1<const Flo64>& a_host_visible_mug,
                            const ViewUHL1<const Flo64>& a_host_visible_vparg,
                            const ViewUHL1<const Flo64>& a_host_visible_coeff_intdmu,
                            const ViewUHL1<const Flo64>& a_host_visible_coeff_intdvpar,
                            const ViewUHL1<const Flo64>& a_host_visible_rg,
                            const ViewUHL1<const Flo64>& a_host_visible_nustar0_r,
                            const ViewUHL1<const Flo64>& a_host_visible_q_r,
                            const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r,
                            const ViewUHL2<const Flo64>& a_host_visible_mask_LIM,
                            const ViewUHL2<const Flo64>& a_host_visible_B_norm,
                            const ViewUHL4<const Flo64>& a_host_visible_Bstar_s)
            // Constant, externally computed values
            : pSpecies_id_{pSpecies_id}
            , mu_id_{mu_id}
            , the_mu_extent_{the_mu_extent}
            , the_vpar_extent_{the_vpar_extent}
            , the_r_extent_{the_r_extent}
            , the_theta_extent_{the_theta_extent}
            , the_phi_extent_{the_phi_extent}
            , the_species_extent_{the_species_extent}
            , R0_{R0}
            , collision_interspecies_{collision_interspecies}
            , SOL_{SOL}
            , LIM_{LIM}
            , ir_SOL_separatrix_{ir_SOL_separatrix}
            , dt_{dt}
            , comb_mat_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "comb_mat_"),
                        2 * kMaximumLaguerrePolynomialDegree, 2 * kMaximumLaguerrePolynomialDegree}
            , hat_As_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_As_"),
                      kMaximumLaguerrePolynomialDegree}
            , hat_Zs_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_Zs_"),
                      kMaximumLaguerrePolynomialDegree}
            , mug_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mug_"),
                   the_mu_extent_}
            , vparg_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "vparg_"),
                     the_vpar_extent_}
            , coeff_intdmu_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "coeff_intdmu_"),
                            the_mu_extent_}
            , coeff_intdvpar_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "coeff_intdvpar_"),
                              the_vpar_extent_}
            , rg_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "rg_"),
                  the_r_extent_}
            , nustar0_r_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar0_r_"),
                         the_r_extent_}
            , q_r_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_r_"),
                   the_r_extent_}
            , mask_buffer_r_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_buffer_r_"),
                             the_r_extent_}
            , mask_LIM_{}
            , B_norm_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "B_norm_"),
                      the_r_extent_, the_theta_extent_ + 1 /* Weird +1 */}
            , Bstar_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "Bstar_s_"),
                       the_vpar_extent_, the_r_extent_, the_theta_extent_, the_species_extent_}

            // Constant, internally computed values

            , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_j_"),
                   kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree}
            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    2 * kMaximumLaguerrePolynomialDegree - 1, kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree}
            , nustar_coeff_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar_coeff_"),
                            the_species_extent_, the_species_extent_, the_r_extent_}

            // Variable, internally computed values

            , n_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_s_"),
                   the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}
            , U_par_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_"),
                       the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}
            , T_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_s_"),
                   the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}
            , nu_ab_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu_ab_"),
                     the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_, the_species_extent_}
            , q_par_on_nT_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT_s_"),
                             the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_}

            // Miscellaneous tooling

            // TODO(Etienne M): Should we partition ?
            , the_alpha_solver_{Kokkos::DefaultExecutionSpace{}} {

            // TODO(Etienne M): Say we get CPU buffers. Depending on whether
            // koliop is compiled for CPU or GPU we can either, do a shallow
            // copy, or we need to do a deep copy across memory spaces.
            // Copy::Shallow tries not to allocate more memory than necessary.
            // Copy::Deep always reallocate.
            // Both Copy::Deep and Copy::Shallow know how to handle views
            // of different layout and memory space.
            koliop::utility::Copy::Shallow(a_host_visible_comb_mat, comb_mat_);
            koliop::utility::Copy::Shallow(a_host_visible_hat_As, hat_As_);
            koliop::utility::Copy::Shallow(a_host_visible_hat_Zs, hat_Zs_);
            koliop::utility::Copy::Shallow(a_host_visible_mug, mug_);
            koliop::utility::Copy::Shallow(a_host_visible_vparg, vparg_);
            koliop::utility::Copy::Shallow(a_host_visible_coeff_intdmu, coeff_intdmu_);
            koliop::utility::Copy::Shallow(a_host_visible_coeff_intdvpar, coeff_intdvpar_);
            koliop::utility::Copy::Shallow(a_host_visible_rg, rg_);
            koliop::utility::Copy::Shallow(a_host_visible_nustar0_r, nustar0_r_);
            koliop::utility::Copy::Shallow(a_host_visible_q_r, q_r_);
            koliop::utility::Copy::Shallow(a_host_visible_mask_buffer_r, mask_buffer_r_);

            if(LIM_) {
                mask_LIM_ = ViewMDD2<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_LIM_"),
                                            the_r_extent, the_theta_extent};
                koliop::utility::Copy::Shallow(a_host_visible_mask_LIM, mask_LIM_);
            }

            koliop::utility::Copy::Shallow(a_host_visible_B_norm, B_norm_);
            koliop::utility::Copy::Shallow(a_host_visible_Bstar_s, Bstar_s_);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();

            DoPrepareCj_Cjl(C_j_, C_jl_);
            DoPrepareNuStar(nustar_coeff_,
                            a_host_visible_hat_As,
                            a_host_visible_hat_Zs,
                            a_host_visible_rg,
                            a_host_visible_nustar0_r,
                            a_host_visible_q_r,
                            a_host_visible_mask_buffer_r);
        }

        template <typename View3D,
                  typename View2D>
        inline void State::DoPrepareCj_Cjl(View2D& C_j,
                                           View3D& C_jl) const {
            // a_host_visible_C_j is zero initialized.
            auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

            for(uint64_t j = 0; j < kMaximumLaguerrePolynomialDegree; ++j) {
                a_host_visible_C_j(0, j) = 1.0;
                a_host_visible_C_j(1, j) = static_cast<Flo64>(-static_cast<int64_t>(j));
            }

            if(kMaximumLaguerrePolynomialDegree > 2) {
                a_host_visible_C_j(2, 2) = 0.5;
                if(kMaximumLaguerrePolynomialDegree > 3) {
                    // NOTE: Assuming kMaximumLaguerrePolynomialDegree = 3, this
                    // is never evaluated.
                    a_host_visible_C_j(2, 3) = 1.5;
                    a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_j, C_j);

            DoPrepareCjl(a_host_visible_C_j, C_jl);
        }

        template <typename View3D,
                  typename MirrorView2D>
        inline void State::DoPrepareCjl(const MirrorView2D& a_host_visible_C_j,
                                        View3D&             C_jl) const {
            // a_host_visible_C_jl is zero initialized.
            auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl_);

            const auto CoreLoop = [&](auto j_tmp, auto l_tmp) {
                for(uint64_t i = 0; i <= (j_tmp + l_tmp); ++i) {
                    // This seems unnecessary ?
                    a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                }

                for(uint64_t alpha = 0; alpha <= j_tmp; ++alpha) {
                    for(uint64_t beta = 0; beta <= l_tmp; ++beta) {
                        a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) += a_host_visible_C_j(alpha, j_tmp) *
                                                                           a_host_visible_C_j(beta, l_tmp);
                    }
                }
            };

            for(uint64_t i = 0; i < kMaximumLaguerrePolynomialDegree; ++i) {
                for(uint64_t j = 0; j < kMaximumLaguerrePolynomialDegree; ++j) {
                    CoreLoop(i, j);
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_jl, C_jl);
        }

        inline void State::DoPrepareNuStar(ViewMDD3<Flo64>&             nustar_coeff,
                                           const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                                           const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                                           const ViewUHL1<const Flo64>& a_host_visible_rg,
                                           const ViewUHL1<const Flo64>& a_host_visible_nustar0_r,
                                           const ViewUHL1<const Flo64>& a_host_visible_q_r,
                                           const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r) const {
            // TODO(Etienne M): I observe some slight numerical discrepancy
            // (1e-14 ~ ) when comparing the result of Gysela and koliop.

            ViewMHL1<Flo64> a_host_visible_mask_space{Kokkos::view_alloc(Kokkos::WithoutInitializing, "a_host_visible_mask_space"),
                                                      the_r_extent_};

            if(a_host_visible_q_r(0) < koliop::portability::GyselaConstant<Flo64>::kInfinity) {
                for(uint64_t i = 0; i < the_r_extent_; ++i) {
                    a_host_visible_mask_space(i) = 1.0 / a_host_visible_q_r(i);
                }
            }

            if(!(LIM_ || SOL_)) {
                for(uint64_t i = 0; i < the_r_extent_; ++i) {
                    a_host_visible_mask_space(i) *= (1.0 - a_host_visible_mask_buffer_r(i));
                }
            }

            const auto a_host_visible_nustar_coeff = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                            Kokkos::WithoutInitializing),
                                                                                         nustar_coeff);

            const auto R0_inverse = 1.0 / R0_;

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
            for(uint64_t ispeciesB = 0; ispeciesB < the_species_extent_; ++ispeciesB) {
                for(uint64_t ispeciesA = 0; ispeciesA < the_species_extent_; ++ispeciesA) {
                    const Flo64 hat_Aa = a_host_visible_hat_As(ispeciesA);
                    const Flo64 hat_Za = a_host_visible_hat_Zs(ispeciesA);
                    const Flo64 hat_Ab = a_host_visible_hat_As(ispeciesB);
                    const Flo64 hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                    const Flo64 AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                            (1.0 / hat_Aa) *
                                            (1.0 / hat_Aa + 1.0 / hat_Ab);

                    for(uint64_t ir = 0; ir < the_r_extent_; ++ir) {
                        // Computation of
                        // AD(r) = sqrt(2) * eps ^ (3 / 2) / (q R0_)*nustaR0_(r)
                        // which is independent on the species considered (eps = r/R0_)
                        const Flo64 eps = a_host_visible_rg(ir) * R0_inverse;

                        const Flo64 AD = a_host_visible_nustar0_r(ir) *
                                         a_host_visible_mask_space(ir) *
                                         // TODO(Etienne M): SIMPLIFY
                                         // sqrt(2)*(rg*R0_inverse)*sqrt(rg*R0_inverse)*R0_inverse_ =
                                         // sqrt(2)*pow(r, 5/2)*pow(rg, 3/2)
                                         // sqrt(2)*pow(r, 5/2) is constant here.
                                         koliop::portability::MathematicalConstant<Flo64>::kSQRT2 *
                                         eps *
                                         Kokkos::sqrt(eps) *
                                         R0_inverse;

                        // Computation of nu_ab =
                        // (AD * hat_Za ^ 2 * hat_Zb ^ 2 * Nb) /
                        // (vTa ^ 2 + vTb ^ 2) ^ (3 / 2) * (1 / hat_Aa) *
                        // (1 / hat_Aa + 1 / hat_Ab)
                        a_host_visible_nustar_coeff(ispeciesA, ispeciesB, ir) = AD * AD_factor;
                    }
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_nustar_coeff, nustar_coeff);
        }


    } // namespace collision
} // namespace koliop

#endif
