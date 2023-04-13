// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        namespace {
            // NOTE: I use an unamed namespace because it is unknown to me if
            // adding static prior KOKKOS_INLINE_FUNCTION is legal.

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

        } // namespace

        template <typename Array6D,
                  typename Array4D,
                  typename Array2D>
        static inline void
        DoComputeAlphaProjection(const Operator& an_operator,
                                 uint64_t        block_first_r,
                                 uint64_t        block_first_theta,
                                 uint64_t        block_first_phi,
                                 const Array2D&  mask_coll,
                                 const Array6D&  f5d_s,
                                 Array4D&&       the_rhs) {

            const uint64_t& the_mu_extent   = an_operator.the_mu_extent_;
            const uint64_t& the_vpar_extent = an_operator.the_vpar_extent_;

            const auto& mug          = an_operator.mug_;
            const auto& coeff_intdmu = an_operator.coeff_intdmu_;
            const auto& B_norm       = an_operator.B_norm_;
            const auto& T_s          = an_operator.T_s_;

            const uint64_t& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t the_block_local_index_r     = ir % Operator::kBlockExtentR;
                const uint64_t the_block_local_index_theta = itheta % Operator::kBlockExtentTheta;
                const uint64_t the_block_local_index_phi   = iphi % Operator::kBlockExtentPhi;

                const uint64_t the_flattened_index = Operator::kBlockExtentR * Operator::kBlockExtentTheta * the_block_local_index_phi +
                                                     Operator::kBlockExtentR * the_block_local_index_theta +
                                                     the_block_local_index_r;

                const uint64_t the_batch_index         = the_flattened_index / Operator::AlphaSolverType::kBatchSize;
                const uint64_t the_batch_element_index = the_flattened_index % Operator::AlphaSolverType::kBatchSize;

                if(mask_coll(ir, itheta)) {
                    const Flo64 BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                    for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                        for(uint64_t ipol = 0; ipol < Operator::kMaximumLaguerrePolynomialDegree; ++ipol) {
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
                        for(uint64_t ipol = 0; ipol < Operator::kMaximumLaguerrePolynomialDegree; ++ipol) {
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
                                                                                             {block_first_r + Operator::kBlockExtentR,
                                                                                              block_first_theta + Operator::kBlockExtentTheta,
                                                                                              block_first_phi + Operator::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeAlphaProjection",
                                 a_policy, the_kernel);
        }

        template <typename Array2D0,
                  typename Array2D1,
                  typename Array1D>
        static inline void
        DoComputeBlockTo1D(const Operator& an_operator,
                           uint64_t        block_first_r,
                           uint64_t        block_first_theta,
                           uint64_t        block_first_phi,
                           const Array2D0& mask_coll,
                           Array2D1&&      n1d_s,
                           Array2D1&&      T1d_s,
                           Array1D&&       U_par1d,
                           Array1D&&       q_par_on_nT1d,
                           Array1D&&       nu1d) {

            const uint64_t& the_species_extent = an_operator.the_species_extent_;

            const auto& n_s           = an_operator.n_s_;
            const auto& U_par_s       = an_operator.U_par_s_;
            const auto& T_s           = an_operator.T_s_;
            const auto& nu_ab         = an_operator.nu_ab_;
            const auto& q_par_on_nT_s = an_operator.q_par_on_nT_s_;

            const uint64_t& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t block_local_index_r     = ir % Operator::kBlockExtentR;
                const uint64_t block_local_index_theta = itheta % Operator::kBlockExtentTheta;
                const uint64_t block_local_index_phi   = iphi % Operator::kBlockExtentPhi;

                const uint64_t ispace = Operator::kBlockExtentR * Operator::kBlockExtentTheta * block_local_index_phi +
                                        Operator::kBlockExtentR * block_local_index_theta +
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
                                                                                             {block_first_r + Operator::kBlockExtentR,
                                                                                              block_first_theta + Operator::kBlockExtentTheta,
                                                                                              block_first_phi + Operator::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeBlockTo1D",
                                 a_policy, the_kernel);
        }

        template <typename Array6D,
                  typename Array4D,
                  typename Array2D>
        static inline void
        DoComputeReconstruction(const Operator& an_operator,
                                uint64_t        block_first_r,
                                uint64_t        block_first_theta,
                                uint64_t        block_first_phi,
                                const Array2D&  mask_coll,
                                Array6D&&       f5d_s,
                                const Array4D&  the_rhs) {

            const uint64_t& the_mu_extent   = an_operator.the_mu_extent_;
            const uint64_t& the_vpar_extent = an_operator.the_vpar_extent_;

            const auto& mug    = an_operator.mug_;
            const auto& B_norm = an_operator.B_norm_;
            const auto& T_s    = an_operator.T_s_;

            const uint64_t& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?
                // TODO(Etienne M): Expensive DoComputeLaguerrePol call.

                // NOTE: kBlockExtentXXX are constant, the compiler can replace
                // the % with faster instructions.
                const uint64_t the_block_local_index_r     = ir % Operator::kBlockExtentR;
                const uint64_t the_block_local_index_theta = itheta % Operator::kBlockExtentTheta;
                const uint64_t the_block_local_index_phi   = iphi % Operator::kBlockExtentPhi;

                const uint64_t the_flattened_index = Operator::kBlockExtentR * Operator::kBlockExtentTheta * the_block_local_index_phi +
                                                     Operator::kBlockExtentR * the_block_local_index_theta +
                                                     the_block_local_index_r;

                const uint64_t the_batch_index         = the_flattened_index / Operator::AlphaSolverType::kBatchSize;
                const uint64_t the_batch_element_index = the_flattened_index % Operator::AlphaSolverType::kBatchSize;

                if(!mask_coll(ir, itheta)) {
                    return;
                }

                const Flo64 BoT = B_norm(ir, itheta) / T_s(ir, itheta, iphi, pSpecies_id);

                for(uint64_t ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(uint64_t imu = 0; imu < the_mu_extent; ++imu) {
                        const Flo64 muBoT       = mug(imu) * BoT;
                        Flo64       f5d_s_value = 0.0;

                        for(uint64_t ipol = 0; ipol < Operator::kMaximumLaguerrePolynomialDegree; ++ipol) {
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
                                                                                             {block_first_r + Operator::kBlockExtentR,
                                                                                              block_first_theta + Operator::kBlockExtentTheta,
                                                                                              block_first_phi + Operator::kBlockExtentPhi}};

            Kokkos::parallel_for("DoComputeReconstruction",
                                 a_policy, the_kernel);
        }

        template <typename Array6D>
        static inline void
        DoApply(const Operator& an_operator,
                Array6D&&       f5d_s) {
            // TODO(Etienne M): The blocking is mainly done for memory consumption reasons,
            // we have to do something similar.

            // TODO(Etienne M): we can drop the ir_start/ir_end and just pass the range as
            // the arrays are not subarrays.

            const uint64_t& the_vpar_extent    = an_operator.the_vpar_extent_;
            const uint64_t& the_r_extent       = an_operator.the_r_extent_;
            const uint64_t& the_theta_extent   = an_operator.the_theta_extent_;
            const uint64_t& the_phi_extent     = an_operator.the_phi_extent_;
            const uint64_t& the_species_extent = an_operator.the_species_extent_;

            const auto& mask_buffer_r = an_operator.mask_buffer_r_;
            const auto& mask_LIM      = an_operator.mask_LIM_;

            const bool& LIM = an_operator.LIM_;

            const auto the_alpha_solver = an_operator.the_alpha_solver_;

            KOLIOP_ASSERT(the_r_extent % Operator::kBlockExtentR == 0);
            KOLIOP_ASSERT(the_theta_extent % Operator::kBlockExtentTheta == 0);
            KOLIOP_ASSERT(the_phi_extent % Operator::kBlockExtentPhi == 0);

            // TODO(Etienne M): Many opportunities to use constant extent say:
            // Flo64 *[Operator::kFlattenedExtent].

            koliop::utility::View::MDD2<bool> mask_coll{Kokkos::view_alloc(Kokkos::WithoutInitializing, "mask_coll"),
                                                        Operator::kBlockExtentR, Operator::kBlockExtentTheta};

            // TODO(Etienne M): Remove the *1D* n, T U Nu etc.
            // Then build a matrix builder which samples the matrix returning
            // the value which we can then cache and use in
            // the_differential_operator and the_rhs. The builder produces the
            // dt/4*T piece of matrix.
            // mask_coll should also be obtain on the fly.

            koliop::utility::View::MDD2<Flo64> n1d_s{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n1d_s"),
                                                     Operator::kFlattenedExtent, the_species_extent};
            koliop::utility::View::MDD2<Flo64> T1d_s{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T1d_s"),
                                                     Operator::kFlattenedExtent, the_species_extent};
            koliop::utility::View::MDD1<Flo64> U_par1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par1d"),
                                                       Operator::kFlattenedExtent};
            koliop::utility::View::MDD1<Flo64> q_par_on_nT1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT1d"),
                                                             Operator::kFlattenedExtent};
            koliop::utility::View::MDD1<Flo64> nu1d{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu1d"),
                                                    Operator::kFlattenedExtent};

            koliop::utility::View::MDD4<Flo64> N0{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N0"),
                                                  the_vpar_extent, Operator::kFlattenedExtent, Operator::kMaximumLaguerrePolynomialDegree, Operator::kMaximumLaguerrePolynomialDegree};
            koliop::utility::View::MDD4<Flo64> N1{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                                                  the_vpar_extent, Operator::kFlattenedExtent, Operator::kMaximumLaguerrePolynomialDegree, Operator::kMaximumLaguerrePolynomialDegree};
            koliop::utility::View::MDD4<Flo64> N2{Kokkos::view_alloc(Kokkos::WithoutInitializing, "N1"),
                                                  the_vpar_extent, Operator::kFlattenedExtent, Operator::kMaximumLaguerrePolynomialDegree, Operator::kMaximumLaguerrePolynomialDegree};

            Operator::AlphaSolverType::ScalarMatrixType the_differential_operator{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                                     "the_differential_operator"),
                                                                                  Operator::kFlattenedExtent, the_vpar_extent};

            Operator::AlphaSolverType::ScalarVectorType the_rhs{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                                   "the_rhs"),
                                                                Operator::kFlattenedExtent, the_vpar_extent};

            // TODO(Etienne M): On GPU, should we just scale the block size ?
            // TODO(Etienne M): We use mask to avoid some computation, but
            // wouldn't it be netter to compute for everything and only in the
            // end, reconstruct f based on the mask ?

            for(uint64_t block_first_phi = 0;
                block_first_phi < the_phi_extent;
                block_first_phi += Operator::kBlockExtentPhi) {
                for(uint64_t block_first_theta = 0;
                    block_first_theta < the_theta_extent;
                    block_first_theta += Operator::kBlockExtentTheta) {
                    for(uint64_t block_first_r = 0;
                        block_first_r < the_r_extent;
                        block_first_r += Operator::kBlockExtentR) {

                        const uint64_t block_last_r     = block_first_r + Operator::kBlockExtentR;
                        const uint64_t block_last_theta = block_first_theta + Operator::kBlockExtentTheta;

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
                            DoComputeAlphaProjection(an_operator,
                                                     block_first_r,
                                                     block_first_theta,
                                                     block_first_phi,
                                                     mask_coll, f5d_s, the_rhs);

                            // IN: Constants upon entering the function.
                            // OUT: n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d
                            DoComputeBlockTo1D(an_operator,
                                               block_first_r,
                                               block_first_theta,
                                               block_first_phi,
                                               mask_coll, n1d_s, T1d_s, U_par1d, q_par_on_nT1d, nu1d);

                            // IN: matrix
                            // OUT: matrix
                            the_alpha_solver.Factorize(the_differential_operator.data(), Operator::kFlattenedExtent, the_vpar_extent);

                            // IN: matrix, RHS_vec
                            // OUT: RHS_vec
                            the_alpha_solver.SolveVector(the_differential_operator.data(), Operator::kFlattenedExtent, the_vpar_extent,
                                                         the_rhs.data());

                            // IN: matrix, RHS_vec
                            // OUT: RHS_vec
                            the_alpha_solver.SolveVector(the_differential_operator.data(), Operator::kFlattenedExtent, the_vpar_extent,
                                                         the_rhs.data());

                            DoComputeReconstruction(an_operator,
                                                    block_first_r,
                                                    block_first_theta,
                                                    block_first_phi,
                                                    mask_coll, f5d_s, the_rhs);
                        }
                    }
                }
            }
        }

        void Operator::Apply(Flo64* f5d_s) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = koliop::utility::View::MDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                                             the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = koliop::utility::View::UHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                                           f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoApply(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
