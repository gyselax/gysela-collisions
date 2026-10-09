// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>

#include "KOLIOP/collisions.h"
#include "KOLIOP/collisions/constants.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/iterate.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collisions {

        /// Correction to keep the local density constant
        /// After the main part of the collisions, the distribution function has
        ///  evolved:
        ///   C(Fa) = Fa'
        /// The new density na' can be different than the one before collisions
        /// In order to keep the local density constant one can applied a correction
        ///  after collisions by taking Fa'' = Fa' * na /na'
        ///
        template <typename ExecutionSpace,
                  typename View6D>
        static inline void
        DoComputeDensityCorrection(const Operator& an_operator,
                                   ExecutionSpace  an_execution_space,
                                   View6D&&        f5d_s) {

            const auto& mu_extent      = an_operator.mu_extent_;
            const auto& vpar_extent    = an_operator.vpar_extent_;
            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;

            const auto team_count = tor1_extent * tor2_extent * tor3_extent * species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            const auto a_policy = PolicyType{
                an_execution_space,
                static_cast<int>(team_count),
                // NOTE: See the note in the DoComputeMoments kernel
                // regarding thread block size.
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64, 64)};

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType itor1     = (team_index) % tor1_extent;
                const OrdinalType itor2     = (team_index / (tor1_extent)) % tor2_extent;
                const OrdinalType itor3     = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                const OrdinalType ispeciesA = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                ComputeFloatType              na;
                Kokkos::Sum<ComputeFloatType> na_reducer{na};
                na_reducer.init(na_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadRange +
                // explicit na_reducer followed by team_reduce, a valid
                // semantic.

                using koliop::utility::IntraTeam;
                IntraTeam::Iterate2D(
                    mu_extent, vpar_extent,
                    a_team_handle, [&](auto imu, auto ivpar) {
                        // NOTE: This is not vectorized on CPU due to the
                        // reduction.

                        const ComputeFloatType f5d_s_value = Kokkos::max(f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA),
                                                                         0.0);

                        const ComputeFloatType ftmp = Bstar_s(ivpar, itor2, itor1, ispeciesA) *
                                                      coeff_intdmu(imu) *
                                                      coeff_intdvpar(ivpar) *
                                                      f5d_s_value;

                        na_reducer.join(na_reducer.reference(),
                                        ftmp);

                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = f5d_s_value;
                    });

                a_team_handle.team_reduce(na_reducer);

                // NOTE: This should be a no-op.
                na = na_reducer.reference();

                na *= koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                const ComputeFloatType na_inverse = 1.0 / na;

                IntraTeam::Iterate2D(
                    mu_extent, vpar_extent,
                    a_team_handle, [&](auto imu, auto ivpar) {
                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) *= n_s(itor1, itor2, itor3, ispeciesA) * na_inverse;
                    });
            };

            Kokkos::parallel_for("DoComputeDensityCorrection",
                                 a_policy, kernel);
        }

        void Operator::ComputeDensityCorrection(const koliop::utility::View::MDL6<Flo64>& f5d_s) const {
            KOLIOP_COLLISIONS_CHECK_DISTRIBUTION_EXTENTS(f5d_s);

            DoComputeDensityCorrection(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                       f5d_s);
        }

    } // namespace collisions
} // namespace koliop
