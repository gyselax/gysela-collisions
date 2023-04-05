// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

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
        template <typename Float>
        using ViewMDL6 = Kokkos::View<Float******, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;

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

        template <typename Float>
        using ViewMHL1 = Kokkos::View<Float*, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace::memory_space>;


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
        protected:
            static constexpr uint64_t kBlockExtentR     = 8;
            static constexpr uint64_t kBlockExtentTheta = 1;
            static constexpr uint64_t kBlockExtentPhi   = 1;

            static constexpr uint64_t kFlattenedExtent = kBlockExtentR * kBlockExtentTheta * kBlockExtentPhi;

        public:
            State(uint64_t                     Npolmax,
                  uint64_t                     index_max,
                  uint64_t                     pSpecies_id,
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

        protected:
            void DoPrepareCj_Cjl(ViewMDD2<Flo64>& C_j,
                                 ViewMDD3<Flo64>& C_jl) const;

            void DoPrepareCjl(const ViewMDD2<Flo64>::host_mirror_type& a_host_visible_C_j,
                              ViewMDD3<Flo64>&                         C_jl) const;

            void DoPrepareNuStar(ViewMDD3<Flo64>&             nustar_coeff,
                                 const ViewUHL1<const Flo64>& a_host_visible_hat_As,
                                 const ViewUHL1<const Flo64>& a_host_visible_hat_Zs,
                                 const ViewUHL1<const Flo64>& a_host_visible_rg,
                                 const ViewUHL1<const Flo64>& a_host_visible_nustar0_r,
                                 const ViewUHL1<const Flo64>& a_host_visible_q_r,
                                 const ViewUHL1<const Flo64>& a_host_visible_mask_buffer_r) const;

            template <typename Array6D,
                      typename Array4D>
            static void
            DoComputeMoments(const State& a_state,
                             Array6D&&    f5d_s,
                             Array4D&&    n_s,
                             Array4D&&    U_par_s,
                             Array4D&&    T_s);

            static KOKKOS_INLINE_FUNCTION Flo64
            DoComputeFrequencyNuab(Flo64 nustar_coeff,
                                   Flo64 hat_As_speciesA,
                                   Flo64 hat_As_speciesB,
                                   Flo64 Ta,
                                   Flo64 nb,
                                   Flo64 Tb);

            template <typename Array5D>
            static void
            DoPrecomputeNu(const State& a_state,
                           Array5D&&    nu_ab);

            template <typename Array6D,
                      typename Array4D>
            static void
            DoComputeEvolutionMaxwellian(const State& a_state,
                                         Array6D&&    f5d_s,
                                         Array4D&&    T_s);

            template <typename Array6D,
                      typename Array4D>
            static void
            DoComputeQparOnNT(const State&   a_state,
                              const Array6D& f5d_s,
                              Array4D&&      q_par_on_nT_s);

            template <typename Array6D>
            static void
            DoComputeCpar(const State& a_state,
                          Array6D&&    f5d_s);

            template <typename Array6D>
            static void
            DoComputeOperationMaxwellianUnshiftedLoc(const State& a_state,
                                                     Array6D&&    f5d_s,
                                                     uint32_t     choice);

            template <typename Array6D>
            static void
            DoApplyCollisions(const State& a_state,
                              Array6D&&    f5d_s);

            template <typename Array6D,
                      typename Array3D,
                      typename Array2D>
            static void
            DoComputeAlphaProjection(const State&   a_state,
                                     uint64_t       block_first_r,
                                     uint64_t       block_first_theta,
                                     uint64_t       block_first_phi,
                                     const Array2D& mask_coll,
                                     const Array6D& f5d_s,
                                     Array3D&&      RHS_vec);

            void DoComputeBlockTo1d(uint64_t                       ir_block_size,
                                    uint64_t                       itheta_block_size,
                                    uint64_t                       iphi_block_size,
                                    const ViewUHL2<const uint8_t>& a_host_visible_mask_coll,
                                    const ViewUHL4<const Flo64>&   a_host_visible_n_s,
                                    const ViewUHL4<const Flo64>&   a_host_visible_U_par_s,
                                    const ViewUHL4<const Flo64>&   a_host_visible_T_s,
                                    const ViewUHL4<const Flo64>&   a_host_visible_q_par_on_nT_s,
                                    ViewUHL2<Flo64>&&              a_host_visible_n1d_s,
                                    ViewUHL2<Flo64>&&              a_host_visible_T1d_s,
                                    ViewUHL1<Flo64>&&              a_host_visible_U_par1d,
                                    ViewUHL1<Flo64>&&              a_host_visible_q_par_on_nT1d,
                                    ViewUHL1<Flo64>&&              a_host_visible_nu1d) const;

            void DoComputeReconstruction(uint64_t                       ir_block_size,
                                         uint64_t                       itheta_block_size,
                                         uint64_t                       iphi_block_size,
                                         const ViewUHL2<const uint8_t>& a_host_visible_mask_coll,
                                         const ViewUHL2<const Flo64>&   a_host_visible_B_norm,
                                         ViewUHL6<Flo64>&&              a_host_visible_f5d_s,
                                         const ViewUHL4<const Flo64>&   a_host_visible_T_s,
                                         const ViewUHL3<const Flo64>&   a_host_visible_RHS_vec) const;

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

            static KOKKOS_INLINE_FUNCTION Flo64
            DoComputeLaguerrePol(uint64_t l,
                                 Flo64    x);

            // Constant, externally computed values

            // TODO(Etienne M): Cleanup unused values (say, Nspecies repalced with the_species_extent_)
            uint64_t Npolmax_;
            uint64_t index_max_;
            uint64_t Nspecies_;
            uint64_t pSpecies_id_;
            uint64_t mu_id_;
            uint64_t Nmu_;
            uint64_t Nvpar_;
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

            ViewMDD2<Flo64> comb_mat_;
            ViewMDD1<Flo64> hat_As_;
            ViewMDD1<Flo64> hat_Zs_;
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

            ViewMDD2<Flo64> C_j_;
            ViewMDD3<Flo64> C_jl_;
            ViewMDD3<Flo64> nustar_coeff_;

            // Variable, internally computed values

            ViewMDD4<Flo64> n_s_;
            ViewMDD4<Flo64> U_par_s_;
            ViewMDD4<Flo64> T_s_;
            ViewMDD5<Flo64> nu_ab_;
            ViewMDD4<Flo64> q_par_on_nT_s_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

        inline State::State(uint64_t                     Npolmax,
                            uint64_t                     index_max,
                            uint64_t                     pSpecies_id,
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
            : // Constant, externally computed values

            Npolmax_{Npolmax}
            , index_max_{index_max}
            , Nspecies_{the_species_extent}
            , pSpecies_id_{pSpecies_id}
            , mu_id_{mu_id}
            , Nmu_{the_mu_extent - 1}
            , Nvpar_{the_vpar_extent - 1}
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
                        ((index_max_ + 1) + 1) - 0, ((index_max_ + 1) + 1) - 0}
            , hat_As_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_As_"),
                      Npolmax_}
            , hat_Zs_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "hat_Zs_"),
                      Npolmax_}
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

            // TODO(Etienne M): Compile time code based on Npolmax_ value ?
            , C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_j_"),
                   Npolmax_, Npolmax_}
            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    index_max_ + 1, Npolmax_, Npolmax_}
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
                             the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_} {

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

        inline void State::ComputeMoments(Flo64* f5d_s) {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeMoments(*this, a_device_visible_f5d_s, n_s_, U_par_s_, T_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::PrecomputeNu() {
            DoPrecomputeNu(*this, nu_ab_);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeEvolutionMaxwellian(Flo64* f5d_s) {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeEvolutionMaxwellian(*this, a_device_visible_f5d_s, T_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeQparOnNT(const Flo64* f5d_s) {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<const Flo64>{// "unmanaged_host_f5d_s",
                                                              f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeQparOnNT(*this, a_device_visible_f5d_s, q_par_on_nT_s_);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeCpar(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeCpar(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeOperationMaxwellianUnshiftedLoc(Flo64*   f5d_s,
                                                                  uint32_t choice) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeOperationMaxwellianUnshiftedLoc(*this, a_device_visible_f5d_s, choice);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ApplyCollisions(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoApplyCollisions(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeCorrectionDensity(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeCorrectionDensity(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::ComputeCorrectionUparT(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, Nspecies_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeCorrectionUparT(*this, a_device_visible_f5d_s, U_par_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

        inline void State::DoPrepareCj_Cjl(ViewMDD2<Flo64>& C_j,
                                           ViewMDD3<Flo64>& C_jl) const {
            // a_host_visible_C_j is zero initialized.
            auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

            for(uint64_t j = 0; j < Npolmax_; ++j) {
                a_host_visible_C_j(0, j) = 1.0;
                a_host_visible_C_j(1, j) = static_cast<Flo64>(-static_cast<int64_t>(j));
            }

            if(Npolmax_ > 2) {
                a_host_visible_C_j(2, 2) = 0.5;
                if(Npolmax_ > 3) {
                    a_host_visible_C_j(2, 3) = 1.5;
                    a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_j, C_j);

            DoPrepareCjl(a_host_visible_C_j, C_jl);
        }

        inline void State::DoPrepareCjl(const ViewMDD2<Flo64>::host_mirror_type& a_host_visible_C_j,
                                        ViewMDD3<Flo64>&                         C_jl) const {
            // a_host_visible_C_jl is zero initialized.
            auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl_);

            const auto CoreLoop = [&](auto j_tmp, auto l_tmp) {
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

            for(uint64_t i = 0; i < Npolmax_; ++i) {
                for(uint64_t j = 0; j < Npolmax_; ++j) {
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
            for(uint64_t ispeciesB = 0; ispeciesB < Nspecies_; ++ispeciesB) {
                for(uint64_t ispeciesA = 0; ispeciesA < Nspecies_; ++ispeciesA) {
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

        template <typename Array6D,
                  typename Array4D>
        void State::DoComputeMoments(const State& a_state,
                                     Array6D&&    f5d_s,
                                     Array4D&&    n_s,
                                     Array4D&&    U_par_s,
                                     Array4D&&    T_s) {

            const auto& the_mu_extent      = a_state.the_mu_extent_;
            const auto& the_vpar_extent    = a_state.the_vpar_extent_;
            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.Nspecies_;

            const auto& mug            = a_state.mug_;
            const auto& vparg          = a_state.vparg_;
            const auto& coeff_intdmu   = a_state.coeff_intdmu_;
            const auto& coeff_intdvpar = a_state.coeff_intdvpar_;
            const auto& B_norm         = a_state.B_norm_;
            const auto& Bstar_s        = a_state.Bstar_s_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi, auto ispecies) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                Flo64 na    = 0.0;
                Flo64 vpara = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 f5d_s_value = Kokkos::max(1.0E-40,
                                                              f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const Flo64 ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                           (coeff_intdmu(imu) * f5d_s_value);
                        na += ftmp;
                        vpara += vparg(ivpar) * ftmp;

                        f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) = f5d_s_value;
                    }
                }

                vpara = vpara / na;

                Flo64 Ta = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        // TODO(Etienne M): Avoid this recomputation ? On
                        // Gysela fortran, a temporary buffer is used but
                        // that would hardly fit on a GPU.
                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const Flo64 ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                           (coeff_intdmu(imu) * f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));
                        Ta += ((vparg(ivpar) - vpara) * (vparg(ivpar) - vpara) +
                               2.0 * mug(imu) * B_norm(ir, itheta)) *
                              ftmp;
                    }
                }

                Ta = Ta / (3.0 * na);
                na = koliop::portability::MathematicalConstant<Flo64>::k2Pi * na;

                n_s(ir, itheta, iphi, ispecies)     = na;
                U_par_s(ir, itheta, iphi, ispecies) = vpara;
                T_s(ir, itheta, iphi, ispecies)     = Ta;
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent,
                                                                                              the_species_extent}};

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, the_kernel);
        }

        KOKKOS_INLINE_FUNCTION Flo64
        State::DoComputeFrequencyNuab(Flo64 nustar_coeff,
                                      Flo64 hat_As_speciesA,
                                      Flo64 hat_As_speciesB,
                                      Flo64 Ta,
                                      Flo64 nb,
                                      Flo64 Tb) {

            const Flo64 vTa = Kokkos::sqrt(Ta / hat_As_speciesA);
            const Flo64 vTb = Kokkos::sqrt(Tb / hat_As_speciesB);

            // TODO(Etienne M): SIMPLIFY We could do:
            // (nustar_coeff * nb) *
            //     std::pow((Tb * hat_As_speciesA +
            //               Ta * hat_As_speciesB) /
            //                  (hat_As_speciesA * hat_As_speciesB),
            //              -3.0 / 2.0);
            return (nustar_coeff * nb) / ((vTa * vTa + vTb * vTb) * Kokkos::sqrt(vTa * vTa + vTb * vTb));
        }

        template <typename Array5D>
        void State::DoPrecomputeNu(const State& a_state,
                                   Array5D&&    nu_ab) {

            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.Nspecies_;

            const auto& hat_As       = a_state.hat_As_;
            const auto& nustar_coeff = a_state.nustar_coeff_;
            const auto& n_s          = a_state.n_s_;
            const auto& T_s          = a_state.T_s_;

            const auto& ir_SOL_separatrix = a_state.ir_SOL_separatrix_;

            const auto the_kernel = KOKKOS_LAMBDA(auto itheta, auto iphi, auto ispeciesA, auto ispeciesB) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const uint64_t the_local_r_extent = Kokkos::min(the_r_extent, ir_SOL_separatrix + 1);

                for(uint64_t ir = 0; ir < the_local_r_extent; ++ir) {
                    nu_ab(ir, itheta, iphi, ispeciesA, ispeciesB) = DoComputeFrequencyNuab(nustar_coeff(ispeciesA, ispeciesB, ir),
                                                                                           hat_As(ispeciesA), hat_As(ispeciesB),
                                                                                           T_s(ir, itheta, iphi, ispeciesA),
                                                                                           n_s(ir, itheta, iphi, ispeciesB),
                                                                                           T_s(ir, itheta, iphi, ispeciesB));
                }

                for(uint64_t ir = the_local_r_extent; ir < the_r_extent; ++ir) {
                    // Propagate the value at nu_ab(ir_SOL_separatrix_, itheta, iphi, ispeciesA, ispeciesB) or
                    // nu_ab(the_local_r_extent - 1, itheta, iphi, ispeciesA, ispeciesB)
                    nu_ab(ir, itheta, iphi, ispeciesA, ispeciesB) = nu_ab(ir_SOL_separatrix, itheta, iphi, ispeciesA, ispeciesB);
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0},
                                                                                             {the_theta_extent,
                                                                                              the_phi_extent,
                                                                                              the_species_extent,
                                                                                              the_species_extent}};

            Kokkos::parallel_for("DoPrecomputeNu",
                                 a_policy, the_kernel);
        }

        template <typename Array6D,
                  typename Array4D>
        void State::DoComputeEvolutionMaxwellian(const State& a_state,
                                                 Array6D&&    f5d_s,
                                                 Array4D&&    T_s) {

            const auto& the_mu_extent      = a_state.the_mu_extent_;
            const auto& the_vpar_extent    = a_state.the_vpar_extent_;
            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.Nspecies_;

            const auto& hat_As  = a_state.hat_As_;
            const auto& mug     = a_state.mug_;
            const auto& vparg   = a_state.vparg_;
            const auto& B_norm  = a_state.B_norm_;
            const auto& n_s     = a_state.n_s_;
            const auto& U_par_s = a_state.U_par_s_;
            const auto& nu_ab   = a_state.nu_ab_;

            const Flo64 dt = a_state.dt_ / 2.0;

            // TODO(Etienne M): I wish we had MDRange with scratch memory.
            // Instead of having to use a TeamPolicy and computing MDRange index
            // by hand.

            // NOTE: We use scratch memory to avoid overwriting the T_s.
            using ScratchMemory = Kokkos::View<Flo64*, Kokkos::DefaultExecutionSpace::scratch_memory_space>;

            const uint64_t the_scratch_memory_bytes_per_thread = the_species_extent * sizeof(Flo64);

            using KernelIterationPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;

            const uint64_t the_team_count = the_theta_extent * the_phi_extent;
            // Due to the inherent cuda/HIP threadblock limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {
                // Reference to the_kernel's member variable copy.
                const auto the_kernel_core = [&](auto ir, auto itheta, auto iphi, auto& Ta_tmp) {
                    // NOTE: Iterating on species leads to large strides in
                    // memory access. This is mitigated on GPU but painful on
                    // CPU. This behavior done to avoid copying the whole Ts_.
                    // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                    for(uint64_t ispeciesA = 0; ispeciesA < the_species_extent; ++ispeciesA) {
                        const Flo64 hat_Aa = hat_As(ispeciesA);
                        const Flo64 na     = n_s(ir, itheta, iphi, ispeciesA);
                        const Flo64 Vpara  = U_par_s(ir, itheta, iphi, ispeciesA);
                        const Flo64 Ta     = T_s(ir, itheta, iphi, ispeciesA);

                        Flo64 Ta_new = Ta;

                        // TODO(Etienne M): Split into two loops to avoid the if ?

                        for(uint64_t ispeciesB = 0; ispeciesB < the_species_extent; ++ispeciesB) {
                            if(ispeciesA == ispeciesB) {
                                continue;
                            }

                            const Flo64 hat_Ab = hat_As(ispeciesB);
                            const Flo64 Vparb  = U_par_s(ir, itheta, iphi, ispeciesB);
                            const Flo64 Tb     = T_s(ir, itheta, iphi, ispeciesB);

                            const Flo64 nu = nu_ab(ir, itheta, iphi, ispeciesA, ispeciesB);

                            Ta_new += dt * nu * (2.0 * hat_Aa * (Tb - Ta) / (hat_Aa + hat_Ab) + (2.0 / 3.0) * Vpara * (Vpara - sqrt(hat_Aa / hat_Ab) * Vparb));
                        }

                        // We can not overwrite the T_s until all the new_T_s are
                        // computed.
                        Ta_tmp(ispeciesA) = Ta_new;

                        const Flo64 coeff_old = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta *
                                                      Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));
                        const Flo64 coeff_new = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta_new *
                                                      Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta_new));
                        const Flo64 Bij       = B_norm(ir, itheta);
                        const Flo64 bot_old   = Bij / Ta;
                        const Flo64 bot_new   = Bij / Ta_new;

                        for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            const Flo64 vp2_old = (vparg(ivpar) - Vpara) * (vparg(ivpar) - Vpara) / (2.0 * Ta);
                            const Flo64 vp2_new = (vparg(ivpar) - Vpara) * (vparg(ivpar) - Vpara) / (2.0 * Ta_new);
                            for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                                f5d_s(imu, ivpar, ir, itheta, iphi, ispeciesA) -= coeff_old * Kokkos::exp(-bot_old * mug(imu) - vp2_old) +
                                                                                  coeff_new * Kokkos::exp(-bot_new * mug(imu) - vp2_new);
                            }
                        }
                    }

                    for(uint64_t ispeciesA = 0; ispeciesA < the_species_extent; ++ispeciesA) {
                        T_s(ir, itheta, iphi, ispeciesA) = Ta_tmp(ispeciesA);
                    }
                };

                const auto the_team_index = a_team_handle.league_rank();

                const auto iphi   = the_team_index / the_theta_extent;
                const auto itheta = the_team_index % the_theta_extent;

                // TODO(Etienne M): Should be put that view allocation in the
                // parallel_for below ? I dont think so as it would lead to
                // bumping the LDS pointer for each R iteration. I think this
                // should be done once per thread.
                // TODO(Etienne M): Should we a_team_handle.team_barrier() ?
                ScratchMemory the_thread_scratch_memory{a_team_handle.team_scratch(0),
                                                        the_scratch_memory_bytes_per_thread};

                const auto a_policy = Kokkos::TeamThreadRange(a_team_handle,
                                                              the_r_extent);

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

        template <typename Array6D,
                  typename Array4D>
        void State::DoComputeQparOnNT(const State&   a_state,
                                      const Array6D& f5d_s,
                                      Array4D&&      q_par_on_nT_s) {

            const auto& the_mu_extent      = a_state.the_mu_extent_;
            const auto& the_vpar_extent    = a_state.the_vpar_extent_;
            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.Nspecies_;

            const auto& mug            = a_state.mug_;
            const auto& vparg          = a_state.vparg_;
            const auto& coeff_intdmu   = a_state.coeff_intdmu_;
            const auto& coeff_intdvpar = a_state.coeff_intdvpar_;
            const auto& B_norm         = a_state.B_norm_;
            const auto& Bstar_s        = a_state.Bstar_s_;
            const auto& n_s            = a_state.n_s_;
            const auto& U_par_s        = a_state.U_par_s_;
            const auto& T_s            = a_state.T_s_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi, auto ispecies) {
                // TODO(Etienne M): Should we flatten even more the loops ?
                // Potentially computing/loading redundant values (Bij, na etc.)
                // but extracting more parallelisms. The thing is that ivpar and
                // imu tends to be larger than ir, itheta or iphi.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 Bij   = B_norm(ir, itheta);
                const Flo64 na    = n_s(ir, itheta, iphi, ispecies);
                const Flo64 Vpara = U_par_s(ir, itheta, iphi, ispecies);
                const Flo64 Ta    = T_s(ir, itheta, iphi, ispecies);
                // TODO(Etienne M): SIMPLIFY as std::pow(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta, -3 / 2);
                const Flo64 coeff = 1.0 / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));

                Flo64 q_par_on_nT_s_reduction = 0.0;

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const Flo64 jac = koliop::portability::MathematicalConstant<Flo64>::k2Pi *
                                      Bstar_s(ivpar, ir, itheta, ispecies);

                    const Flo64 vparg_minus_vpara         = vparg(ivpar) - Vpara;
                    const Flo64 vparg_minus_vpara_squared = vparg_minus_vpara * vparg_minus_vpara;
                    const Flo64 vp2                       = vparg_minus_vpara_squared / (2.0 * Ta);

                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 u_tmp   = mug(imu) * Bij / Ta;
                        const Flo64 FMa_tmp = na * Kokkos::exp(-u_tmp - vp2) * coeff;
                        const Flo64 v2      = 2.0 * mug(imu) * Bij + vparg_minus_vpara_squared;
                        q_par_on_nT_s_reduction += 0.5 *
                                                   jac *
                                                   coeff_intdmu(imu) *
                                                   coeff_intdvpar(ivpar) *
                                                   vparg_minus_vpara *
                                                   v2 *
                                                   (f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) - FMa_tmp);
                    }
                }

                q_par_on_nT_s(ir, itheta, iphi, ispecies) = q_par_on_nT_s_reduction / (na * Ta);
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent,
                                                                                              the_species_extent}};

            Kokkos::parallel_for("DoComputeQparOnNT",
                                 a_policy, the_kernel);
        }

        template <typename Array6D>
        void State::DoComputeCpar(const State& a_state,
                                  Array6D&&    f5d_s) {

            const auto& the_mu_extent      = a_state.the_mu_extent_;
            const auto& the_vpar_extent    = a_state.the_vpar_extent_;
            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.Nspecies_;

            const auto& hat_As        = a_state.hat_As_;
            const auto& hat_Zs        = a_state.hat_Zs_;
            const auto& mug           = a_state.mug_;
            const auto& vparg         = a_state.vparg_;
            const auto& rg            = a_state.rg_;
            const auto& nustar0_r     = a_state.nustar0_r_;
            const auto& q_r           = a_state.q_r_;
            const auto& B_norm        = a_state.B_norm_;
            const auto& n_s           = a_state.n_s_;
            const auto& U_par_s       = a_state.U_par_s_;
            const auto& T_s           = a_state.T_s_;
            const auto& q_par_on_nT_s = a_state.q_par_on_nT_s_;

            const auto  pSpecies_id = a_state.pSpecies_id_;
            const Flo64 R0_inverse  = 1.0 / a_state.R0_;
            const Flo64 dt          = a_state.dt_ / 2.0;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): Assuming materialization, we could flatten
                // the loop even more, up to Nmu_.
                // TODO(Etienne M): We could inter exchange some loops ?
                // TODO(Etienne M): There is quite a lot to simplify here.
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                const Flo64 hat_Aa     = hat_As(pSpecies_id);
                const Flo64 hat_Za     = hat_Zs(pSpecies_id);
                const Flo64 inv_hat_Aa = 1.0 / hat_Aa;

                const Flo64 smcsfqri = q_r(ir);
                const Flo64 Bij      = B_norm(ir, itheta);

                const Flo64 na          = n_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 Vpara       = U_par_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 Ta          = T_s(ir, itheta, iphi, pSpecies_id);
                const Flo64 q_par_on_nT = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);

                const Flo64 nustaR0_i = nustar0_r(ir);
                const Flo64 epstemp   = rg(ir) * R0_inverse;
                const Flo64 vTa       = Kokkos::sqrt(Ta / hat_Aa);
                const Flo64 coeff1    = 1.0 / (2.0 * Ta);
                const Flo64 coeff3    = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));
                const Flo64 AD        = nustaR0_i * koliop::portability::MathematicalConstant<Flo64>::kSQRT2 * epstemp * Kokkos::sqrt(epstemp) / ((1.0 / R0_inverse) * (smcsfqri != koliop::portability::GyselaConstant<Flo64>::kInfinity ? smcsfqri : 1.0));

                for(uint64_t ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    const Flo64 hat_Ab         = hat_As(ispecies);
                    const Flo64 inv_hat_Ab     = 1.0 / hat_Ab;
                    const Flo64 hat_Zb         = hat_Zs(ispecies);
                    const Flo64 nb             = n_s(ir, itheta, iphi, ispecies);
                    const Flo64 Vparb          = U_par_s(ir, itheta, iphi, ispecies);
                    const Flo64 Tb             = T_s(ir, itheta, iphi, ispecies);
                    const Flo64 q_parb_on_nbTb = q_par_on_nT_s(ir, itheta, iphi, ispecies);

                    const Flo64 Rat    = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const Flo64 vTb    = Kokkos::sqrt(Tb / hat_Ab);
                    const Flo64 coeff2 = vTa / vTb;
                    const Flo64 coeff4 = (3.0 / 5.0) * Rat * q_parb_on_nbTb / (1.0 + (Ta * hat_Ab) / (Tb * hat_Aa));

                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            const Flo64 coeff5         = (vparg(ivpar) / Ta);
                            const Flo64 xa             = Kokkos::sqrt(coeff1 * (2.0 * mug(imu) * Bij + vparg(ivpar) * vparg(ivpar)));
                            const Flo64 xb             = xa * coeff2;
                            const Flo64 FM0a           = coeff3 * Kokkos::exp(-xa * xa);
                            const Flo64 Theta_func_tmp = xb > 1E-15 ?
                                                             koliop::portability::GyselaConstant<Flo64>::kConst_theta * ((Kokkos::erf(xb) - xb * (koliop::portability::GyselaConstant<Flo64>::kCst_phiprime * exp(-xb * xb))) / (2.0 * xb * xb)) / xb :
                                                             1.0;
                            const Flo64 nu_sab         = (AD * hat_Za * hat_Za * hat_Zb * hat_Zb * nb) * inv_hat_Aa * (inv_hat_Aa + inv_hat_Ab) * Theta_func_tmp / (vTb * vTb * vTb);

                            f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) += dt * FM0a * nu_sab * coeff5 * (Rat * Vparb - Vpara + q_par_on_nT * (1.0 - 2.0 * xa * xa * 0.2) - coeff4);
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeCpar",
                                 a_policy, the_kernel);
        }

        template <typename Array6D>
        void State::DoComputeOperationMaxwellianUnshiftedLoc(const State& a_state,
                                                             Array6D&&    f5d_s,
                                                             uint32_t     choice) {

            const auto& the_mu_extent    = a_state.the_mu_extent_;
            const auto& the_vpar_extent  = a_state.the_vpar_extent_;
            const auto& the_r_extent     = a_state.the_r_extent_;
            const auto& the_theta_extent = a_state.the_theta_extent_;
            const auto& the_phi_extent   = a_state.the_phi_extent_;

            const auto& mug    = a_state.mug_;
            const auto& vparg  = a_state.vparg_;
            const auto& B_norm = a_state.B_norm_;
            const auto& n_s    = a_state.n_s_;
            const auto& T_s    = a_state.T_s_;

            const auto pSpecies_id = a_state.pSpecies_id_;

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            // TODO(Etienne M): We could use a single kernel but know that
            // flattening means recomputation, and thus, due to the slight
            // discrepancies in the two kernels below, we would need many
            // branches. Should we keep 2 flattened kernels ?

            switch(choice) {
                case 3: {
                    const auto the_kernel_choice_3 = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                        // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                        const Flo64 Bij    = B_norm(ir, itheta);
                        const Flo64 na     = n_s(ir, itheta, iphi, pSpecies_id);
                        const Flo64 Ta     = T_s(ir, itheta, iphi, pSpecies_id);
                        const Flo64 BijoTa = Bij / Ta;
                        const Flo64 f2oTa  = 0.5 / Ta;
                        const Flo64 coeff  = na / (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta));

                        for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                                const Flo64 v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = Kokkos::max(0.0,
                                                                                               coeff * Kokkos::exp(-v2oTa) * f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id));
                            }
                        }
                    };

                    Kokkos::parallel_for("DoComputeOperationMaxwellianUnshiftedLoc3",
                                         a_policy, the_kernel_choice_3);
                    break;
                }
                case 4: {
                    const auto the_kernel_choice_4 = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                        // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                        const Flo64 Bij    = B_norm(ir, itheta);
                        const Flo64 na     = n_s(ir, itheta, iphi, pSpecies_id);
                        const Flo64 Ta     = T_s(ir, itheta, iphi, pSpecies_id);
                        const Flo64 BijoTa = Bij / Ta;
                        const Flo64 f2oTa  = 0.5 / Ta;
                        const Flo64 coeff  = (koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<Flo64>::k2Pi * Ta)) / na;
                        for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                                const Flo64 v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = Kokkos::max(0.001,
                                                                                               Kokkos::min(1000.0,
                                                                                                           coeff * exp(v2oTa) * f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id)));
                            }
                        }
                    };

                    Kokkos::parallel_for("DoComputeOperationMaxwellianUnshiftedLoc4",
                                         a_policy, the_kernel_choice_4);
                    break;
                }
                default:
                    // TODO(Etienne M): Should this case be reachable ?
                    KOLIOP_ASSERT(false);
                    break;
            }
        }

        template <typename Array6D>
        void State::DoApplyCollisions(const State& a_state,
                                      Array6D&&    f5d_s) {
            // TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
            // we have to do something similar.

            // TODO(Etienne M): we can drop the ir_start/ir_end and just pass the range as
            // the arrays are not subarrays.

            const auto& the_vpar_extent    = a_state.the_vpar_extent_;
            const auto& the_r_extent       = a_state.the_r_extent_;
            const auto& the_theta_extent   = a_state.the_theta_extent_;
            const auto& the_phi_extent     = a_state.the_phi_extent_;
            const auto& the_species_extent = a_state.the_species_extent_;

            const auto& mask_buffer_r = a_state.mask_buffer_r_;
            const auto& mask_LIM      = a_state.mask_LIM_;

            const auto& Npolmax = a_state.Npolmax_;
            const auto& LIM     = a_state.LIM_;

            KOLIOP_ASSERT(the_r_extent % kBlockExtentR == 0);
            KOLIOP_ASSERT(the_theta_extent % kBlockExtentTheta == 0);
            KOLIOP_ASSERT(the_phi_extent % kBlockExtentPhi == 0);

            // TODO(Etienne M): Many opportunities to use constant extent say:
            // Flo64 *[KFlattenedExtent].

            ViewMDD2<bool> mask_coll{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll"),
                                     kBlockExtentR, kBlockExtentTheta};

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

            ViewMDD3<Flo64> RHS_vec{Kokkos::view_alloc(Kokkos::WithoutInitializing, "RHS_vec"),
                                    kFlattenedExtent, Npolmax, the_vpar_extent};
            ViewMDD4<Flo64> N0{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N0"),
                               the_vpar_extent, kFlattenedExtent, Npolmax, Npolmax};
            ViewMDD4<Flo64> N1{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                               the_vpar_extent, kFlattenedExtent, Npolmax, Npolmax};
            ViewMDD4<Flo64> N2{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                               the_vpar_extent, kFlattenedExtent, Npolmax, Npolmax};
            ViewMDD3<Flo64> D_mat{Kokkos::view_alloc(Kokkos::WithoutInitializing, "D_mat"),
                                  kFlattenedExtent, Npolmax, Npolmax * the_vpar_extent};
            ViewMDD3<Flo64> DL_mat{Kokkos::view_alloc(Kokkos::WithoutInitializing, "DL_mat"),
                                   kFlattenedExtent, Npolmax, Npolmax * (the_vpar_extent - 1)}; // TODO(Etienne M): Weird -1
            ViewMDD3<Flo64> DU1_mat{Kokkos::view_alloc(Kokkos::WithoutInitializing, "DU1_mat"),
                                    kFlattenedExtent, Npolmax, Npolmax * (the_vpar_extent - 1)}; // TODO(Etienne M): Weird -1
            ViewMDD3<Flo64> DU2_mat{Kokkos::view_alloc(Kokkos::WithoutInitializing, "DU2_mat"),
                                    kFlattenedExtent, Npolmax, Npolmax * (the_vpar_extent - 2)}; // TODO(Etienne M): Weird -2

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
                        const uint64_t block_last_phi   = block_first_phi + kBlockExtentPhi;

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

                        // TODO(Etienne M): Basically if we have at least one
                        // bit to 1, we do not skip.
                        // if ( count(mask_coll).eq.0 ) cycle

                        DoComputeAlphaProjection(a_state,
                                                 block_first_r,
                                                 block_first_theta,
                                                 block_first_phi,
                                                 mask_coll, f5d_s, RHS_vec);

                        // DoComputeBlockTo1d(ir_start, ir_end,
                        //                    itheta_start, itheta_end,
                        //                    iphi_start, iphi_end,
                        //                    mask_coll, n1d_s, U_par1d, T1d_s, q_par_on_nT1d, nu1d);

                        // DoComputeCvCd(n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d,
                        //               RHS_vec, N0, N1, N2, D_mat, DL_mat, DU1_mat, DU2_mat);

                        // DoComputeReconstruction(ir_start, ir_end,
                        //                         itheta_start, itheta_end,
                        //                         iphi_start, iphi_end,
                        //                         mask_coll, f5d_s, RHS_vec);
                    }
                }
            }
        }

        template <typename Array6D,
                  typename Array3D,
                  typename Array2D>
        void State::DoComputeAlphaProjection(const State&   a_state,
                                             uint64_t       block_first_r,
                                             uint64_t       block_first_theta,
                                             uint64_t       block_first_phi,
                                             const Array2D& mask_coll,
                                             const Array6D& f5d_s,
                                             Array3D&&      RHS_vec) {

            const auto& the_mu_extent   = a_state.the_mu_extent_;
            const auto& the_vpar_extent = a_state.the_vpar_extent_;

            const auto& mug          = a_state.mug_;
            const auto& coeff_intdmu = a_state.coeff_intdmu_;
            const auto& B_norm       = a_state.B_norm_;
            const auto& T_s          = a_state.T_s_;

            const auto& Npolmax = a_state.Npolmax_;

            const auto pSpecies_id = a_state.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t block_local_index_r     = ir % kBlockExtentR;
                const uint64_t block_local_index_theta = itheta % kBlockExtentTheta;
                const uint64_t block_local_index_phi   = iphi % kBlockExtentPhi;

                const uint64_t ispace = kBlockExtentR * kBlockExtentTheta * block_local_index_phi +
                                        kBlockExtentR * block_local_index_theta +
                                        block_local_index_r;

                if(mask_coll(ir, itheta)) {
                    const Flo64 BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                    for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(uint64_t ipol = 0; ipol < Npolmax; ++ipol) {
                            Flo64 RHS_vec_value = 0.0;
                            for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                                const Flo64 muBoT     = mug(imu) * BoT;
                                const Flo64 coeff_tot = DoComputeLaguerrePol(ipol, muBoT) * BoT * coeff_intdmu(imu) * Kokkos::exp(-muBoT);

                                RHS_vec_value += coeff_tot * f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id);
                            }

                            RHS_vec(ispace, ipol, ivpar) = RHS_vec_value;
                        }
                    }
                } else {
                    for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(uint64_t ipol = 0; ipol < Npolmax; ++ipol) {
                            RHS_vec(ispace, ipol, ivpar) = 0.0;
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

        inline void State::DoComputeBlockTo1d(uint64_t                       ir_block_size,
                                              uint64_t                       itheta_block_size,
                                              uint64_t                       iphi_block_size,
                                              const ViewUHL2<const uint8_t>& a_host_visible_mask_coll,
                                              const ViewUHL4<const Flo64>&   a_host_visible_n_s,
                                              const ViewUHL4<const Flo64>&   a_host_visible_U_par_s,
                                              const ViewUHL4<const Flo64>&   a_host_visible_T_s,
                                              const ViewUHL4<const Flo64>&   a_host_visible_q_par_on_nT_s,
                                              ViewUHL2<Flo64>&&              a_host_visible_n1d_s,
                                              ViewUHL2<Flo64>&&              a_host_visible_T1d_s,
                                              ViewUHL1<Flo64>&&              a_host_visible_U_par1d,
                                              ViewUHL1<Flo64>&&              a_host_visible_q_par_on_nT1d,
                                              ViewUHL1<Flo64>&&              a_host_visible_nu1d) const {

            const auto a_host_visible_hat_As = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                      Kokkos::WithoutInitializing),
                                                                                   hat_As_);

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // NOTE: INDEX are ideal for both CPU and GPU.

                const uint64_t ispace = iphi * itheta_block_size * ir_block_size +
                                        itheta * ir_block_size +
                                        ir;

                if(!a_host_visible_mask_coll(ir, itheta))
                    return;

                for(uint64_t ispecies = 0; ispecies < Nspecies_; ++ispecies) {
                    a_host_visible_n1d_s(ispace, ispecies) = a_host_visible_n_s(ir, itheta, iphi, ispecies);
                    a_host_visible_T1d_s(ispace, ispecies) = a_host_visible_T_s(ir, itheta, iphi, ispecies);
                }

                a_host_visible_U_par1d(ispace)       = a_host_visible_U_par_s(ir, itheta, iphi, pSpecies_id_);
                a_host_visible_q_par_on_nT1d(ispace) = a_host_visible_q_par_on_nT_s(ir, itheta, iphi, pSpecies_id_);

                a_host_visible_nu1d(ispace) = nu_ab_(ir, itheta, iphi, pSpecies_id_, pSpecies_id_);
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0},
                                                                                           {ir_block_size,
                                                                                            itheta_block_size,
                                                                                            iphi_block_size}};

            Kokkos::parallel_for("DoComputeBlockTo1d",
                                 a_policy, the_kernel);
        }

        inline void State::DoComputeReconstruction(uint64_t                       ir_block_size,
                                                   uint64_t                       itheta_block_size,
                                                   uint64_t                       iphi_block_size,
                                                   const ViewUHL2<const uint8_t>& a_host_visible_mask_coll,
                                                   const ViewUHL2<const Flo64>&   a_host_visible_B_norm,
                                                   ViewUHL6<Flo64>&&              a_host_visible_f5d_s,
                                                   const ViewUHL4<const Flo64>&   a_host_visible_T_s,
                                                   const ViewUHL3<const Flo64>&   a_host_visible_RHS_vec) const {

            const auto a_host_visible_mug = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                                                   Kokkos::WithoutInitializing),
                                                                                mug_);

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                const uint64_t ispace = iphi * itheta_block_size * ir_block_size +
                                        itheta * ir_block_size +
                                        ir;

                if(!a_host_visible_mask_coll(ir, itheta))
                    return;

                const Flo64 BoT = a_host_visible_B_norm(ir, itheta) / a_host_visible_T_s(ir, itheta, iphi, pSpecies_id_);

                for(uint64_t ivpar = 0; ivpar <= Nvpar_; ++ivpar) {
                    // TODO(Etienne M): Interchange loops and remove this
                    // initialization. Use a temporary scalar, and do the
                    // reduction.
                    for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                        a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_) = 0.0;
                    }

                    for(uint64_t ipol = 0; ipol < Npolmax_; ++ipol) {
                        for(uint64_t imu = 0; imu <= Nmu_; ++imu) {
                            const Flo64 muBoT = a_host_visible_mug(imu) * BoT;
                            a_host_visible_f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id_) += a_host_visible_RHS_vec(ispace, ipol, ivpar) *
                                                                                                DoComputeLaguerrePol(ipol, muBoT);
                        }
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>,
                                                        Kokkos::DefaultHostExecutionSpace>{{0, 0, 0},
                                                                                           {ir_block_size,
                                                                                            itheta_block_size,
                                                                                            iphi_block_size}};

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
            const auto& the_species_extent = a_state.Nspecies_;

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

    } // namespace collision
} // namespace koliop

#endif
