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

        protected:
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

            void ComputeEvolutionMaxwellian(Flo64* f5d_s);

            void ComputeQparOnNT(const Flo64* f5d_s);

            void ComputeCpar(Flo64* f5d_s) const;

            void ComputeOperationMaxwellianUnshiftedLoc(Flo64*   f5d_s,
                                                        uint32_t choice) const;

            void ApplyCollisions(Flo64* f5d_s) const;

            void ComputeCorrectionDensity(Flo64* f5d_s) const;

            void ComputeCorrectionUparT(Flo64* f5d_s) const;

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

            template <typename Array6D>
            static void
            DoApplyCollisions(const State& a_state,
                              Array6D&&    f5d_s);

            static KOKKOS_INLINE_FUNCTION Flo64
            DoComputeLaguerrePol(uint64_t l,
                                 Flo64    x);

            template <typename Array6D,
                      typename Array4D,
                      typename Array2D>
            static void
            DoComputeAlphaProjection(const State&   a_state,
                                     uint64_t       block_first_r,
                                     uint64_t       block_first_theta,
                                     uint64_t       block_first_phi,
                                     const Array2D& mask_coll,
                                     const Array6D& f5d_s,
                                     Array4D&&      the_rhs);

            template <typename Array2D0,
                      typename Array2D1,
                      typename Array1D>
            static void
            DoComputeBlockTo1D(const State&    a_state,
                               uint64_t        block_first_r,
                               uint64_t        block_first_theta,
                               uint64_t        block_first_phi,
                               const Array2D0& mask_coll,
                               Array2D1&&      n1d_s,
                               Array2D1&&      T1d_s,
                               Array1D&&       U_par1d,
                               Array1D&&       q_par_on_nT1d,
                               Array1D&&       nu1d);

            template <typename Array6D,
                      typename Array4D,
                      typename Array2D>
            static void
            DoComputeReconstruction(const State&   a_state,
                                    uint64_t       block_first_r,
                                    uint64_t       block_first_theta,
                                    uint64_t       block_first_phi,
                                    const Array2D& mask_coll,
                                    Array6D&&      f5d_s,
                                    const Array4D& the_rhs);

            template <typename Array6D>
            static void
            DoComputeCorrectionDensity(const State& a_state,
                                       Array6D&&    f5d_s);

            template <typename Array6D,
                      typename Array4D>
            static void
            DoComputeCorrectionUparT(const State& a_state,
                                     Array6D&&    f5d_s,
                                     Array4D&&    U_par_s);

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

        inline void State::ApplyCollisions(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoApplyCollisions(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeCorrectionDensity(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeCorrectionDensity(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeCorrectionUparT(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeCorrectionUparT(*this, a_device_visible_f5d_s, U_par_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
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

        template <typename Array6D>
        void State::DoApplyCollisions(const State& a_state,
                                      Array6D&&    f5d_s) {
            // TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
            // we have to do something similar.

            // TODO(Etienne M): we can drop the ir_start/ir_end and just pass the range as
            // the arrays are not subarrays.

            const uint64_t& the_vpar_extent    = a_state.the_vpar_extent_;
            const uint64_t& the_r_extent       = a_state.the_r_extent_;
            const uint64_t& the_theta_extent   = a_state.the_theta_extent_;
            const uint64_t& the_phi_extent     = a_state.the_phi_extent_;
            const uint64_t& the_species_extent = a_state.the_species_extent_;

            const auto& mask_buffer_r = a_state.mask_buffer_r_;
            const auto& mask_LIM      = a_state.mask_LIM_;

            const bool& LIM = a_state.LIM_;

            const auto the_alpha_solver = a_state.the_alpha_solver_;

            KOLIOP_ASSERT(the_r_extent % kBlockExtentR == 0);
            KOLIOP_ASSERT(the_theta_extent % kBlockExtentTheta == 0);
            KOLIOP_ASSERT(the_phi_extent % kBlockExtentPhi == 0);

            // TODO(Etienne M): Many opportunities to use constant extent say:
            // Flo64 *[KFlattenedExtent].

            ViewMDD2<bool> mask_coll{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll"),
                                     kBlockExtentR, kBlockExtentTheta};

            // TODO(Etienne M): Remove the *1D* n, T U Nu etc.
            // Then build a matrix builder which samples the matrix returning
            // the value which we can then cache and use in
            // the_differential_operator and the_rhs. The builder produces the
            // dt/4*T piece of matrix.
            // mask_coll should also be obtain on the fly.

            ViewMDD2<Flo64> n1d_s{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n1d_s"),
                                  kFlattenedExtent, the_species_extent};
            ViewMDD2<Flo64> T1d_s{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T1d_s"),
                                  kFlattenedExtent, the_species_extent};
            ViewMDD1<Flo64> U_par1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par1d"),
                                    kFlattenedExtent};
            ViewMDD1<Flo64> q_par_on_nT1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT1d"),
                                          kFlattenedExtent};
            ViewMDD1<Flo64> nu1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu1d"),
                                 kFlattenedExtent};

            ViewMDD4<Flo64> N0{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N0"),
                               the_vpar_extent, kFlattenedExtent, kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree};
            ViewMDD4<Flo64> N1{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                               the_vpar_extent, kFlattenedExtent, kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree};
            ViewMDD4<Flo64> N2{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                               the_vpar_extent, kFlattenedExtent, kMaximumLaguerrePolynomialDegree, kMaximumLaguerrePolynomialDegree};

            AlphaSolverType::ScalarMatrixType the_differential_operator{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                           "the_differential_operator"),
                                                                        kFlattenedExtent, the_vpar_extent};

            AlphaSolverType::ScalarVectorType the_rhs{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                         "the_rhs"),
                                                      kFlattenedExtent, the_vpar_extent};

            // TODO(Etienne M): On GPU, should we just scale the block size ?
            // TODO(Etienne M): We use mask to avoid some computation, but
            // wouldn't it be netter to compute for everything and only in the
            // end, reconstruct f based on the mask ?

            for(uint64_t block_first_phi = 0;
                block_first_phi < the_phi_extent;
                block_first_phi += kBlockExtentPhi) {
                for(uint64_t block_first_theta = 0;
                    block_first_theta < the_theta_extent;
                    block_first_theta += kBlockExtentTheta) {
                    for(uint64_t block_first_r = 0;
                        block_first_r < the_r_extent;
                        block_first_r += kBlockExtentR) {

                        const uint64_t block_last_r     = block_first_r + kBlockExtentR;
                        const uint64_t block_last_theta = block_first_theta + kBlockExtentTheta;

                        for(uint64_t itheta = block_first_theta; itheta < block_last_theta; ++itheta) {
                            for(uint64_t ir = block_first_r; ir < block_last_r; ++ir) {
                                mask_coll(ir - block_first_r, itheta - block_first_theta) = mask_buffer_r(ir) < 0.99;
                            }
                        }

                        if(LIM) {
                            for(uint64_t itheta = block_first_theta; itheta < block_last_theta; ++itheta) {
                                for(uint64_t ir = block_first_r; ir < block_last_r; ++ir) {
                                    mask_coll(ir - block_first_r, itheta - block_first_theta) |= mask_LIM(ir, itheta) < 0.99;
                                }
                            }
                        }

                        // NOTE: Skipping would create imbalance
                        // if(all boolean in mask_coll are false) {
                        //     continue;
                        // }

                        {
                            // IN: Constants upon entering the function.
                            // OUT: the_rhs
                            DoComputeAlphaProjection(a_state,
                                                     block_first_r,
                                                     block_first_theta,
                                                     block_first_phi,
                                                     mask_coll, f5d_s, the_rhs);

                            // IN: Constants upon entering the function.
                            // OUT: n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d
                            DoComputeBlockTo1D(a_state,
                                               block_first_r,
                                               block_first_theta,
                                               block_first_phi,
                                               mask_coll, n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d);

                            // IN: matrix
                            // OUT: matrix
                            the_alpha_solver.Factorize(the_differential_operator.data(), kFlattenedExtent, the_vpar_extent);

                            // IN: matrix, RHS_vec
                            // OUT: RHS_vec
                            the_alpha_solver.SolveVector(the_differential_operator.data(), kFlattenedExtent, the_vpar_extent,
                                                         the_rhs.data());

                            // IN: matrix, RHS_vec
                            // OUT: RHS_vec
                            the_alpha_solver.SolveVector(the_differential_operator.data(), kFlattenedExtent, the_vpar_extent,
                                                         the_rhs.data());

                            DoComputeReconstruction(a_state,
                                                    block_first_r,
                                                    block_first_theta,
                                                    block_first_phi,
                                                    mask_coll, f5d_s, the_rhs);
                        }
                    }
                }
            }
        }

        KOKKOS_INLINE_FUNCTION Flo64
        State::DoComputeLaguerrePol(uint64_t l, Flo64 x) {
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

        template <typename Array6D,
                  typename Array4D,
                  typename Array2D>
        void State::DoComputeAlphaProjection(const State&   a_state,
                                             uint64_t       block_first_r,
                                             uint64_t       block_first_theta,
                                             uint64_t       block_first_phi,
                                             const Array2D& mask_coll,
                                             const Array6D& f5d_s,
                                             Array4D&&      the_rhs) {

            const uint64_t& the_mu_extent   = a_state.the_mu_extent_;
            const uint64_t& the_vpar_extent = a_state.the_vpar_extent_;

            const auto& mug          = a_state.mug_;
            const auto& coeff_intdmu = a_state.coeff_intdmu_;
            const auto& B_norm       = a_state.B_norm_;
            const auto& T_s          = a_state.T_s_;

            const uint64_t& pSpecies_id = a_state.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t the_block_local_index_r     = ir % kBlockExtentR;
                const uint64_t the_block_local_index_theta = itheta % kBlockExtentTheta;
                const uint64_t the_block_local_index_phi   = iphi % kBlockExtentPhi;

                const uint64_t the_flattened_index = kBlockExtentR * kBlockExtentTheta * the_block_local_index_phi +
                                                     kBlockExtentR * the_block_local_index_theta +
                                                     the_block_local_index_r;

                const uint64_t the_batch_index         = the_flattened_index / AlphaSolverType::kBatchSize;
                const uint64_t the_batch_element_index = the_flattened_index % AlphaSolverType::kBatchSize;

                if(mask_coll(ir, itheta)) {
                    const Flo64 BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                    for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(uint64_t ipol = 0; ipol < kMaximumLaguerrePolynomialDegree; ++ipol) {
                            for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                                const Flo64 muBoT     = mug(imu) * BoT;
                                const Flo64 coeff_tot = DoComputeLaguerrePol(ipol, muBoT) * BoT * coeff_intdmu(imu) * Kokkos::exp(-muBoT);

                                the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) += coeff_tot *
                                                                                                  f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                            }
                        }
                    }
                } else {
                    for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(uint64_t ipol = 0; ipol < kMaximumLaguerrePolynomialDegree; ++ipol) {
                            the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) = 0.0;
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{block_first_r,
                                                                                              block_first_theta,
                                                                                              block_first_phi},
                                                                                             {block_first_r + kBlockExtentR,
                                                                                              block_first_theta + kBlockExtentTheta,
                                                                                              block_first_phi + kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeAlphaProjection",
                                 a_policy, the_kernel);
        }

        template <typename Array2D0,
                  typename Array2D1,
                  typename Array1D>
        void State::DoComputeBlockTo1D(const State&    a_state,
                                       uint64_t        block_first_r,
                                       uint64_t        block_first_theta,
                                       uint64_t        block_first_phi,
                                       const Array2D0& mask_coll,
                                       Array2D1&&      n1d_s,
                                       Array2D1&&      T1d_s,
                                       Array1D&&       U_par1d,
                                       Array1D&&       q_par_on_nT1d,
                                       Array1D&&       nu1d) {

            const uint64_t& the_species_extent = a_state.the_species_extent_;

            const auto& n_s           = a_state.n_s_;
            const auto& U_par_s       = a_state.U_par_s_;
            const auto& T_s           = a_state.T_s_;
            const auto& nu_ab         = a_state.nu_ab_;
            const auto& q_par_on_nT_s = a_state.q_par_on_nT_s_;

            const uint64_t& pSpecies_id = a_state.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t block_local_index_r     = ir % kBlockExtentR;
                const uint64_t block_local_index_theta = itheta % kBlockExtentTheta;
                const uint64_t block_local_index_phi   = iphi % kBlockExtentPhi;

                const uint64_t ispace = kBlockExtentR * kBlockExtentTheta * block_local_index_phi +
                                        kBlockExtentR * block_local_index_theta +
                                        block_local_index_r;

                if(!mask_coll(ir, itheta)) {
                    return;
                }

                for(uint64_t ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    n1d_s(ispace, ispecies) = n_s(ir, itheta, iphi, ispecies);
                    T1d_s(ispace, ispecies) = T_s(ir, itheta, iphi, ispecies);
                }

                U_par1d(ispace)       = U_par_s(ir, itheta, iphi, pSpecies_id);
                q_par_on_nT1d(ispace) = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);
                nu1d(ispace)          = nu_ab(ir, itheta, iphi, pSpecies_id, pSpecies_id);
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{block_first_r,
                                                                                              block_first_theta,
                                                                                              block_first_phi},
                                                                                             {block_first_r + kBlockExtentR,
                                                                                              block_first_theta + kBlockExtentTheta,
                                                                                              block_first_phi + kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeBlockTo1D",
                                 a_policy, the_kernel);
        }

        template <typename Array6D,
                  typename Array4D,
                  typename Array2D>
        void State::DoComputeReconstruction(const State&   a_state,
                                            uint64_t       block_first_r,
                                            uint64_t       block_first_theta,
                                            uint64_t       block_first_phi,
                                            const Array2D& mask_coll,
                                            Array6D&&      f5d_s,
                                            const Array4D& the_rhs) {

            const uint64_t& the_mu_extent   = a_state.the_mu_extent_;
            const uint64_t& the_vpar_extent = a_state.the_vpar_extent_;

            const auto& mug    = a_state.mug_;
            const auto& B_norm = a_state.B_norm_;
            const auto& T_s    = a_state.T_s_;

            const uint64_t& pSpecies_id = a_state.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t the_block_local_index_r     = ir % kBlockExtentR;
                const uint64_t the_block_local_index_theta = itheta % kBlockExtentTheta;
                const uint64_t the_block_local_index_phi   = iphi % kBlockExtentPhi;

                const uint64_t the_flattened_index = kBlockExtentR * kBlockExtentTheta * the_block_local_index_phi +
                                                     kBlockExtentR * the_block_local_index_theta +
                                                     the_block_local_index_r;

                const uint64_t the_batch_index         = the_flattened_index / AlphaSolverType::kBatchSize;
                const uint64_t the_batch_element_index = the_flattened_index % AlphaSolverType::kBatchSize;

                if(!mask_coll(ir, itheta)) {
                    return;
                }

                const Flo64 BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 muBoT       = mug(imu) * BoT;
                        Flo64       f5d_s_value = 0.0;

                        for(uint64_t ipol = 0; ipol < kMaximumLaguerrePolynomialDegree; ++ipol) {
                            f5d_s_value += the_rhs(the_batch_index, ivpar, ipol, the_batch_element_index) *
                                           DoComputeLaguerrePol(ipol, muBoT);
                        }

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = f5d_s_value;
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{block_first_r,
                                                                                              block_first_theta,
                                                                                              block_first_phi},
                                                                                             {block_first_r + kBlockExtentR,
                                                                                              block_first_theta + kBlockExtentTheta,
                                                                                              block_first_phi + kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeReconstruction",
                                 a_policy, the_kernel);
        }

        template <typename Array6D>
        void State::DoComputeCorrectionDensity(const State& a_state,
                                               Array6D&&    f5d_s) {

            const auto& the_mu_extent    = a_state.the_mu_extent_;
            const auto& the_vpar_extent  = a_state.the_vpar_extent_;
            const auto& the_r_extent     = a_state.the_r_extent_;
            const auto& the_theta_extent = a_state.the_theta_extent_;
            const auto& the_phi_extent   = a_state.the_phi_extent_;

            const auto& coeff_intdmu   = a_state.coeff_intdmu_;
            const auto& coeff_intdvpar = a_state.coeff_intdvpar_;
            const auto& Bstar_s        = a_state.Bstar_s_;
            const auto& n_s            = a_state.n_s_;

            const auto pSpecies_id = a_state.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                Flo64 na = 0.0;
                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jacob_vel = koliop::portability::MathematicalConstant<Flo64>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id);
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 ftmp = Kokkos::max(f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id),
                                                       0.0);

                        na += jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp;

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = ftmp;
                    }
                }

                const Flo64 na_inverse = 1.0 / na;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) *= n_s(ir, itheta, iphi, pSpecies_id) * na_inverse;
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeCorrectionDensity",
                                 a_policy, the_kernel);
        }

        template <typename Array6D,
                  typename Array4D>
        void State::DoComputeCorrectionUparT(const State& a_state,
                                             Array6D&&    f5d_s,
                                             Array4D&&    U_par_s) {

            const auto& the_mu_extent      = a_state.the_mu_extent_;
            const auto& the_vpar_extent    = a_state.the_vpar_extent_;
            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.the_species_extent_;

            const auto& hat_As         = a_state.hat_As_;
            const auto& mug            = a_state.mug_;
            const auto& vparg          = a_state.vparg_;
            const auto& coeff_intdmu   = a_state.coeff_intdmu_;
            const auto& coeff_intdvpar = a_state.coeff_intdvpar_;
            const auto& B_norm         = a_state.B_norm_;
            const auto& Bstar_s        = a_state.Bstar_s_;
            const auto& n_s            = a_state.n_s_;
            const auto& T_s            = a_state.T_s_;
            const auto& nu_ab          = a_state.nu_ab_;
            const auto& q_par_on_nT_s  = a_state.q_par_on_nT_s_;

            const auto  pSpecies_id            = a_state.pSpecies_id_;
            const auto  collision_interspecies = a_state.collision_interspecies_;
            const Flo64 dt                     = a_state.dt_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 Bij            = B_norm(ir, itheta);
                const Flo64 na             = n_s(ir, itheta, iphi, pSpecies_id);
                Flo64       impulsion_coll = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jacob_vel = koliop::portability::MathematicalConstant<Flo64>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id);
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        impulsion_coll += coeff_intdvpar(ivpar) *
                                          coeff_intdmu(imu) *
                                          jacob_vel *
                                          vparg(ivpar) *
                                          f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                    }
                }

                const Flo64 U_par_coll = impulsion_coll / na;
                Flo64       T_coll     = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jacob_vel = koliop::portability::MathematicalConstant<Flo64>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id) / na;
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        T_coll += coeff_intdvpar(ivpar) *
                                  coeff_intdmu(imu) *
                                  (jacob_vel / 3.0) *
                                  ((vparg(ivpar) - U_par_coll) * (vparg(ivpar) - U_par_coll) +
                                   2 * mug(imu) * Bij) *
                                  f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                    }
                }

                const Flo64 hat_Aa      = (pSpecies_id);
                const Flo64 Vpara       = U_par_s(ir, itheta, iphi, pSpecies_id);
                Flo64       U_par_new   = U_par_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 q_par_on_nT = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 T_new       = T_s(ir, itheta, iphi, pSpecies_id);

                for(uint64_t ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    if((ispecies != pSpecies_id) && collision_interspecies) {
                        const Flo64 hat_Ab         = hat_As(ispecies);
                        const Flo64 Rat            = Kokkos::sqrt(hat_Aa / hat_Ab);
                        const Flo64 Vparb          = U_par_s(ir, itheta, iphi, ispecies);
                        const Flo64 q_parb_on_nbTb = q_par_on_nT_s(ir, itheta, iphi, ispecies);

                        const Flo64 k_ab = 1.0 / (1.0 +
                                                  hat_Aa * T_s(ir, itheta, iphi, ispecies) /
                                                      (hat_Ab * T_s(ir, itheta, iphi, pSpecies_id)));
                        const Flo64 k_ba = 1.0 / (1.0 +
                                                  hat_Ab * T_s(ir, itheta, iphi, pSpecies_id) /
                                                      (hat_Aa * T_s(ir, itheta, iphi, ispecies)));

                        const Flo64 nu = nu_ab(ir, itheta, iphi, pSpecies_id, ispecies);

                        const Flo64 R_ab = -nu * (Vpara - Rat * Vparb -
                                                  0.6 * k_ab * q_par_on_nT +
                                                  Rat * 0.6 * k_ba * q_parb_on_nbTb);
                        U_par_new += dt * R_ab;
                    }
                }

                const Flo64 coeff_Max_coll = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                                   T_coll *
                                                   Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * T_coll));

                const Flo64 coeff_Max_new = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                                  T_new *
                                                  Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * T_new));

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 u_coll = mug(imu) * Bij / T_coll;
                        const Flo64 u_new  = mug(imu) * Bij / T_new;

                        const Flo64 FMa_coll = coeff_Max_coll * Kokkos::exp(-u_coll -
                                                                            (vparg(ivpar) - U_par_coll) *
                                                                                (vparg(ivpar) - U_par_coll) / (2 * T_coll));
                        const Flo64 FMa_new  = coeff_Max_new * Kokkos::exp(-u_new -
                                                                           (vparg(ivpar) - U_par_new) *
                                                                               (vparg(ivpar) - U_par_new) / (2 * T_new));

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) += FMa_new - FMa_coll;
                    }
                }

                U_par_s(ir, itheta, iphi, pSpecies_id) = U_par_new;
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeCorrectionUparT",
                                 a_policy, the_kernel);
        }

    } // namespace collision
} // namespace koliop

#endif
