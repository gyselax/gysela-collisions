// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View6D,
                  typename View4D>
        static inline void
        DoComputeMaxwellianEvolution(const Operator&  an_operator,
                                     ComputeFloatType dt,
                                     View6D&&         f5d_s,
                                     View4D&&         T_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& hat_As  = an_operator.hat_As_;
            const auto& mug     = an_operator.mug_;
            const auto& vparg   = an_operator.vparg_;
            const auto& B_norm  = an_operator.B_norm_;
            const auto& n_s     = an_operator.n_s_;
            const auto& U_par_s = an_operator.U_par_s_;
            const auto& nu_ab   = an_operator.nu_ab_;

            const ComputeFloatType half_dt = dt / 2.0;

            // TODO(Etienne M): I wish we had MDRange with scratch memory.
            // Instead of having to use a TeamPolicy and computing MDRange index
            // by hand.

            // NOTE: We use scratch memory to avoid overwriting the T_s.
            using ScratchMemory = Kokkos::View<ComputeFloatType*, Kokkos::DefaultExecutionSpace::scratch_memory_space>;

            const uint64_t the_scratch_memory_bytes_per_thread = the_species_extent * sizeof(ComputeFloatType);

            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {
                // Reference to the_kernel's member variable copy.
                const auto the_kernel_core = [&](auto ir, auto itheta, auto iphi, auto& Ta_tmp) {
                    // NOTE: Iterating on species leads to large strides in
                    // memory access. This is mitigated on GPU but painful on
                    // CPU. This behavior done to avoid copying the whole Ts_.
                    // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                    for(OrdinalType ispeciesA = 0; ispeciesA < the_species_extent; ++ispeciesA) {
                        const ComputeFloatType hat_Aa = hat_As(ispeciesA);
                        const ComputeFloatType Vpara  = U_par_s(ir, itheta, iphi, ispeciesA);
                        const ComputeFloatType Ta     = T_s(ir, itheta, iphi, ispeciesA);

                        ComputeFloatType Ta_new = Ta;

                        // TODO(Etienne M): Split into two loops to avoid the if ?

                        for(OrdinalType ispeciesB = 0; ispeciesB < the_species_extent; ++ispeciesB) {
                            if(ispeciesA == ispeciesB) {
                                continue;
                            }

                            const ComputeFloatType hat_Ab = hat_As(ispeciesB);
                            const ComputeFloatType Vparb  = U_par_s(ir, itheta, iphi, ispeciesB);
                            const ComputeFloatType Tb     = T_s(ir, itheta, iphi, ispeciesB);

                            const ComputeFloatType nu = nu_ab(ir, itheta, iphi, ispeciesA, ispeciesB);

                            Ta_new += half_dt * nu * (2.0 * hat_Aa * (Tb - Ta) / (hat_Aa + hat_Ab) + (2.0 / 3.0) * Vpara * (Vpara - Kokkos::sqrt(hat_Aa / hat_Ab) * Vparb));
                        }

                        // We can not overwrite the T_s until all the new_T_s
                        // are computed.
                        Ta_tmp(ispeciesA) = Ta_new;

                        const ComputeFloatType na = n_s(ir, itheta, iphi, ispeciesA);

                        const ComputeFloatType coeff_old = na / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta *
                                                                 Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta));
                        const ComputeFloatType coeff_new = na / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta_new *
                                                                 Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta_new));

                        const ComputeFloatType Bij = B_norm(ir, itheta);

                        const ComputeFloatType bot_old = Bij / Ta;
                        const ComputeFloatType bot_new = Bij / Ta_new;

                        for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                            const ComputeFloatType vp2_old = (vparg(ivpar) - Vpara) * (vparg(ivpar) - Vpara) / (2.0 * Ta);
                            const ComputeFloatType vp2_new = (vparg(ivpar) - Vpara) * (vparg(ivpar) - Vpara) / (2.0 * Ta_new);

                            for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                                f5d_s(imu, ivpar, ir, itheta, iphi, ispeciesA) += coeff_new * Kokkos::exp(-bot_new * mug(imu) - vp2_new) -
                                                                                  coeff_old * Kokkos::exp(-bot_old * mug(imu) - vp2_old);
                            }
                        }
                    }

                    for(OrdinalType ispeciesA = 0; ispeciesA < the_species_extent; ++ispeciesA) {
                        T_s(ir, itheta, iphi, ispeciesA) = Ta_tmp(ispeciesA);
                    }
                };

                const auto the_team_index = a_team_handle.league_rank();

                const auto iphi   = the_team_index / the_theta_extent;
                const auto itheta = the_team_index % the_theta_extent;

                // TODO(Etienne M): Should we use a_team_handle.team_barrier()?
                ScratchMemory the_thread_scratch_memory{a_team_handle.thread_scratch(0),
                                                        the_scratch_memory_bytes_per_thread};

                const auto a_policy = Kokkos::TeamThreadRange(a_team_handle,
                                                              the_r_extent);

                Kokkos::parallel_for(a_policy,
                                     // Reference to the_kernel's member variable copy.
                                     [&](auto ir) {
                                         the_kernel_core(ir, itheta, iphi, the_thread_scratch_memory);
                                     });
            };

            const uint64_t the_team_count = the_theta_extent * the_phi_extent;

            // Due to the inherent cuda/HIP threadblock limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            auto a_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(the_team_count,
                                                                              Kokkos::AUTO);

            a_policy.set_scratch_size(0,
                                      Kokkos::PerThread(ScratchMemory::shmem_size(the_scratch_memory_bytes_per_thread)));

            Kokkos::parallel_for("DoComputeMaxwellianEvolution",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeMaxwellianEvolution(ComputeFloatType dt,
                                                  Flo64*           f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoComputeMaxwellianEvolution(*this, dt, a_device_visible_f5d_s, T_s_);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s,
                                           koliop::utility::View::UHL6<Flo64>{f5d_s,
                                                                              static_cast<std::size_t>(the_mu_extent_), static_cast<std::size_t>(the_vpar_extent_),
                                                                              static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                              static_cast<std::size_t>(the_species_extent_)});

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
