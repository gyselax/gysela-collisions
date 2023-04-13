// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"

namespace koliop {
    namespace collision {

        namespace {
            // NOTE: I use an unamed namespace because it is unknown to me if
            // adding static prior KOKKOS_INLINE_FUNCTION is legal.

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

        } // namespace

        /// TODO(Etienne M): We precompute the species' Nu but in some cases, it
        /// may be cheaper to just rematerialize it.
        ///
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
