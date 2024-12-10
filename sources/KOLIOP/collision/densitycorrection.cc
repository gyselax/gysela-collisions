// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

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
            const auto& r_extent       = an_operator.r_extent_;
            const auto& theta_extent   = an_operator.theta_extent_;
            const auto& phi_extent     = an_operator.phi_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

            const auto team_count = r_extent * theta_extent * phi_extent * species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            const auto a_policy = PolicyType{
                an_execution_space,
                static_cast<int>(team_count),
                // NOTE: The thread block size and the Nmu benefit from being
                // close. This is due to the TeamThreadMDRange which
                // parallelizes only the left most index which is Nmu.
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType ir        = (team_index) % r_extent;
                const OrdinalType itheta    = (team_index / (r_extent)) % theta_extent;
                const OrdinalType iphi      = (team_index / (r_extent * theta_extent)) % phi_extent;
                const OrdinalType ispeciesA = (team_index / (r_extent * theta_extent * phi_extent)) /* % species_extent */;

                // NOTE: The explicit Left iteration direction may not benefit
                // the CPUs depending on how that is implemented.
                const auto an_intra_team_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                                            std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                                              mu_extent, vpar_extent);

                ComputeFloatType              na;
                Kokkos::Sum<ComputeFloatType> na_reducer{na};
                na_reducer.init(na_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit na_reducer followed by team_reduce, a valid
                // semantic.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                           Bstar_s(ivpar, ir, itheta, ispeciesA);
                        const ComputeFloatType ftmp = Kokkos::max(f5d_s(imu, ivpar, ir, itheta, iphi, ispeciesA),
                                                                  0.0);

                        na_reducer.join(na_reducer.reference(),
                                        jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp);

                        f5d_s(imu, ivpar, ir, itheta, iphi, ispeciesA) = ftmp;
                    });

                a_team_handle.team_reduce(na_reducer);

                // NOTE: This should be a no-op.
                na = na_reducer.reference();

                const ComputeFloatType na_inverse = 1.0 / na;

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        f5d_s(imu, ivpar, ir, itheta, iphi, ispeciesA) *= n_s(ir, itheta, iphi, ispeciesA) * na_inverse;
                    });
            };

            Kokkos::parallel_for("DoComputeDensityCorrection",
                                 a_policy, kernel);
        }

        void Operator::ComputeDensityCorrection(Flo64* f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       r_extent_, theta_extent_, phi_extent_,
                                                                                                                       species_extent_);

            {
                DoComputeDensityCorrection(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                           a_device_visible_f5d_s);
            }

            koliop::utility::Copy::ViewToRawData<koliop::utility::View::InternalInputL6<Flo64>,
                                                 koliop::utility::View::ExternalInputL6<Flo64>>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                a_device_visible_f5d_s,
                                                                                                f5d_s,
                                                                                                mu_extent_, vpar_extent_,
                                                                                                r_extent_, theta_extent_, phi_extent_,
                                                                                                species_extent_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
