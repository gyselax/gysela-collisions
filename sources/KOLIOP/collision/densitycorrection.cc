// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/flatten.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/tordu.h"
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
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64, 32)};

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType itor1     = (team_index) % tor1_extent;
                const OrdinalType itor2     = (team_index / (tor1_extent)) % tor2_extent;
                const OrdinalType itor3     = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                const OrdinalType ispeciesA = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                ComputeFloatType              na;
                Kokkos::Sum<ComputeFloatType> na_reducer{na};
                na_reducer.init(na_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadRange +
                // explicit na_reducer followed by team_reduce, a valid
                // semantic.

                const koliop::utility::Flatten2 flattened_mu_vpar{mu_extent, vpar_extent};

                Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle, flattened_mu_vpar.forward()), [&](auto flattened_index) {
                    const auto mu_vpar_loop_body = [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                           Bstar_s(ivpar, ir, itheta, ispeciesA);
                        const ComputeFloatType ftmp = Kokkos::max(f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA),
                                                                  0.0);

                        na_reducer.join(na_reducer.reference(),
                                        jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp);

                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = ftmp;
                    };
                    flattened_mu_vpar.Call(flattened_index, mu_vpar_loop_body);
                });

                a_team_handle.team_reduce(na_reducer);

                // NOTE: This should be a no-op.
                na = na_reducer.reference();

                const ComputeFloatType na_inverse = 1.0 / na;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle, flattened_mu_vpar.forward()), [&](auto flattened_index) {
                    const auto mu_vpar_loop_body = [&](auto imu, auto ivpar) {
                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) *= n_s(itor1, itor2, itor3, ispeciesA) * na_inverse;
                    };
                    flattened_mu_vpar.Call(flattened_index, mu_vpar_loop_body);
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
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
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
                                                                                                tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                species_extent_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
