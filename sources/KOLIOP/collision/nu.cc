// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/select.h"

namespace koliop {
    namespace collision {

        namespace {
            // NOTE: I use an unamed namespace because it is unknown to me if
            // adding static prior KOKKOS_INLINE_FUNCTION is legal.

            KOKKOS_INLINE_FUNCTION ComputeFloatType
            DoComputeFrequencyNuab(ComputeFloatType nustar_coeff,
                                   ComputeFloatType hat_As_speciesA,
                                   ComputeFloatType hat_As_speciesB,
                                   ComputeFloatType Ta,
                                   ComputeFloatType nb,
                                   ComputeFloatType Tb) {

                const ComputeFloatType vTa = Ta / hat_As_speciesA;
                const ComputeFloatType vTb = Tb / hat_As_speciesB;

                // TODO(Etienne M): SIMPLIFY We could do:
                // (nustar_coeff * nb) *
                //     std::pow((Tb * hat_As_speciesA +
                //               Ta * hat_As_speciesB) /
                //                  (hat_As_speciesA * hat_As_speciesB),
                //              -3.0 / 2.0);

                // nu_ab = (AD * hat_Za^2 * hat_Zb^2 * Nb) /
                //         (vTa^2 + vTb^2)^(3/2) *
                //         (1/hat_Aa) *
                //         (1/hat_Aa + 1/hat_Ab)
                return (nustar_coeff * nb) / ((vTa + vTb) * Kokkos::sqrt(vTa + vTb));
            }

        } // namespace

        template <typename ExecutionSpace,
                  typename View5D>
        static inline void
        DoPrecomputeNu(const Operator& an_operator,
                       ExecutionSpace  an_execution_space,
                       View5D&&        nu_ab) {

            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& the_local_domain_r_offset = an_operator.the_local_domain_r_offset_;

            const auto& hat_As       = an_operator.hat_As_;
            const auto& nustar_coeff = an_operator.nustar_coeff_;
            const auto& n_s          = an_operator.n_s_;
            const auto& T_s          = an_operator.T_s_;

            // Ensure the ir_SOL_separatrix_ is always only processed by the
            // rank processing the most outer torus region (in R). If there is
            // a rank with the_local_domain_r_offset > ir_SOL_separatrix_ we
            // break the contract.
            KOLIOP_ASSERT(the_local_domain_r_offset <= an_operator.ir_SOL_separatrix_);

            // Here we make sure ir_SOL_separatrix is in the rank's domain, not
            // global domain.
            const auto& ir_SOL_separatrix = an_operator.ir_SOL_separatrix_ - the_local_domain_r_offset;

            const auto the_team_count = the_theta_extent * the_phi_extent * the_species_extent * the_species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            // TODO(Etienne M): TEAM hardcoded size, check for auto tuning
            // opportunities.
            const auto a_policy = PolicyType{
                an_execution_space,
                static_cast<int>(the_team_count),
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64, 32)};

            const OrdinalType the_local_r_extent = Kokkos::min(the_r_extent, ir_SOL_separatrix + 1);

            const auto the_kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType the_team_index = a_team_handle.league_rank();

                const OrdinalType itheta    = (the_team_index) % the_theta_extent;
                const OrdinalType iphi      = (the_team_index / (the_theta_extent)) % the_phi_extent;
                const OrdinalType ispeciesA = (the_team_index / (the_theta_extent * the_phi_extent)) % the_species_extent;
                const OrdinalType ispeciesB = (the_team_index / (the_theta_extent * the_phi_extent * the_species_extent)) /* % the_species_extent */;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle,
                                                             0, the_local_r_extent),
                                     [&](auto ir) {
                                         nu_ab(ir, itheta, iphi, ispeciesA, ispeciesB) = DoComputeFrequencyNuab(nustar_coeff(ispeciesA, ispeciesB, ir),
                                                                                                                hat_As(ispeciesA), hat_As(ispeciesB),
                                                                                                                T_s(ir, itheta, iphi, ispeciesA),
                                                                                                                n_s(ir, itheta, iphi, ispeciesB),
                                                                                                                T_s(ir, itheta, iphi, ispeciesB));
                                     });

                // NOTE: We must ensure that the DoPrecomputeNu0 kernel ended
                // before DoPrecomputeNu1.
                a_team_handle.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle,
                                                             the_local_r_extent, the_r_extent),
                                     [&](auto ir) {
                                         // Propagate the value at nu_ab(ir_SOL_separatrix, itheta, iphi, ispeciesA, ispeciesB) or
                                         // nu_ab(the_local_r_extent - 1, itheta, iphi, ispeciesA, ispeciesB)
                                         nu_ab(ir, itheta, iphi, ispeciesA, ispeciesB) = nu_ab(ir_SOL_separatrix, itheta, iphi, ispeciesA, ispeciesB);
                                     });
            };

            Kokkos::parallel_for("DoPrecomputeNu",
                                 a_policy, the_kernel);
        }

        void Operator::PrecomputeNu() {

            DoPrecomputeNu(*this, the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                           nu_ab_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
