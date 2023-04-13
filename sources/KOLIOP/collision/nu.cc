// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

namespace koliop {
    namespace collision {

        template <typename Array5D>
        static inline void
        DoPrecomputeNu(const Operator& an_operator,
                       Array5D&&       nu_ab) {

            const uint64_t& the_r_extent       = an_operator.the_r_extent_;
            const uint64_t& the_theta_extent   = an_operator.the_theta_extent_;
            const uint64_t& the_phi_extent     = an_operator.the_phi_extent_;
            const uint64_t& the_species_extent = an_operator.the_species_extent_;

            const auto& hat_As       = an_operator.hat_As_;
            const auto& nustar_coeff = an_operator.nustar_coeff_;
            const auto& n_s          = an_operator.n_s_;
            const auto& T_s          = an_operator.T_s_;

            const uint64_t& ir_SOL_separatrix = an_operator.ir_SOL_separatrix_;

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

        void Operator::PrecomputeNu() {
            DoPrecomputeNu(*this, nu_ab_);

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
