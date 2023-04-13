// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename Array6D>
        static inline void
        DoComputeMaxwellianUnshiftedLoc(const Operator& an_operator,
                                        Array6D&&       f5d_s,
                                        uint32_t        choice) {

            const uint64_t& the_mu_extent    = an_operator.the_mu_extent_;
            const uint64_t& the_vpar_extent  = an_operator.the_vpar_extent_;
            const uint64_t& the_r_extent     = an_operator.the_r_extent_;
            const uint64_t& the_theta_extent = an_operator.the_theta_extent_;
            const uint64_t& the_phi_extent   = an_operator.the_phi_extent_;

            const auto& mug    = an_operator.mug_;
            const auto& vparg  = an_operator.vparg_;
            const auto& B_norm = an_operator.B_norm_;
            const auto& n_s    = an_operator.n_s_;
            const auto& T_s    = an_operator.T_s_;

            const uint64_t& pSpecies_id = an_operator.pSpecies_id_;

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

                    Kokkos::parallel_for("DoComputeMaxwellianUnshiftedLoc3",
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

                    Kokkos::parallel_for("DoComputeMaxwellianUnshiftedLoc4",
                                         a_policy, the_kernel_choice_4);
                    break;
                }
                default:
                    // TODO(Etienne M): Should this case be reachable ?
                    KOLIOP_ASSERT(false);
                    break;
            }
        }

        void Operator::ComputeMaxwellianUnshiftedLoc(Flo64*   f5d_s,
                                                     uint32_t choice) const {
            // TODO(Etienne M): COPY redundant.

            auto a_device_visible_f5d_s = koliop::utility::View::MDL6<Flo64>{Kokkos::view_alloc(Kokkos::WithoutInitializing, "f5d_s"),
                                                                             the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            auto a_host_visible_f5d_s = koliop::utility::View::UHL6<Flo64>{// "unmanaged_host_f5d_s",
                                                                           f5d_s, the_mu_extent_, the_vpar_extent_, the_r_extent_, the_theta_extent_, the_phi_extent_, the_species_extent_};

            koliop::utility::Copy::Shallow(a_host_visible_f5d_s, a_device_visible_f5d_s);

            DoComputeMaxwellianUnshiftedLoc(*this, a_device_visible_f5d_s, choice);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s, a_host_visible_f5d_s);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
