// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {

        template <typename Array6D>
        static inline void
        DoComputeOperationMaxwellianUnshiftedLoc(const State& a_state,
                                                 Array6D&&    f5d_s,
                                                 uint32_t     choice) {

            const uint64_t& the_mu_extent    = a_state.the_mu_extent_;
            const uint64_t& the_vpar_extent  = a_state.the_vpar_extent_;
            const uint64_t& the_r_extent     = a_state.the_r_extent_;
            const uint64_t& the_theta_extent = a_state.the_theta_extent_;
            const uint64_t& the_phi_extent   = a_state.the_phi_extent_;

            const auto& mug    = a_state.mug_;
            const auto& vparg  = a_state.vparg_;
            const auto& B_norm = a_state.B_norm_;
            const auto& n_s    = a_state.n_s_;
            const auto& T_s    = a_state.T_s_;

            const uint64_t& pSpecies_id = a_state.pSpecies_id_;

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
                                                                                                           coeff * Kokkos::exp(v2oTa) * f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id)));
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

        void State::ComputeOperationMaxwellianUnshiftedLoc(Flo64*   f5d_s,
                                                           uint32_t choice) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = ViewMDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                          the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = ViewUHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                        f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeOperationMaxwellianUnshiftedLoc(*this, a_device_visible_f5d_s, choice);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
