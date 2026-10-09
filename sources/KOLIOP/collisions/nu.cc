// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>

#include "KOLIOP/collisions.h"
#include "KOLIOP/collisions/constants.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/select.h"

namespace koliop {
    namespace collisions {
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

        template <typename ExecutionSpace,
                  typename View5D>
        static inline void
        DoPrecomputeNu(const Operator& an_operator,
                       ExecutionSpace  an_execution_space,
                       View5D&&        nu_ab) {

            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& hat_As       = an_operator.hat_As_;
            const auto& nustar_coeff = an_operator.nustar_coeff_;
            const auto& n_s          = an_operator.n_s_;
            const auto& T_s          = an_operator.T_s_;

            // TODO(Etienne M): Just use a range policy.
            const auto team_count = tor2_extent * tor3_extent * species_extent * species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            const auto a_policy = PolicyType{
                an_execution_space,
                static_cast<int>(team_count),
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64, 64)};

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType itor2     = (team_index) % tor2_extent;
                const OrdinalType itor3     = (team_index / (tor2_extent)) % tor3_extent;
                const OrdinalType ispeciesA = (team_index / (tor2_extent * tor3_extent)) % species_extent;
                const OrdinalType ispeciesB = (team_index / (tor2_extent * tor3_extent * species_extent)) /* % species_extent */;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle,
                                                             0, tor1_extent),
                                     [&](auto itor1) {
                                         nu_ab(itor1, itor2, itor3, ispeciesA, ispeciesB) = DoComputeFrequencyNuab(nustar_coeff(ispeciesA, ispeciesB, itor2),
                                                                                                                   hat_As(ispeciesA), hat_As(ispeciesB),
                                                                                                                   T_s(itor1, itor2, itor3, ispeciesA),
                                                                                                                   n_s(itor1, itor2, itor3, ispeciesB),
                                                                                                                   T_s(itor1, itor2, itor3, ispeciesB));
                                     });
            };

            Kokkos::parallel_for("DoPrecomputeNu",
                                 a_policy, kernel);
        }

        void Operator::PrecomputeNu() {
            DoPrecomputeNu(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                           nu_ab_);
        }

    } // namespace collisions
} // namespace koliop
