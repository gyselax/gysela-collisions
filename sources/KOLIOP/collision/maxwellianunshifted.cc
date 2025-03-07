// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <stdexcept>
#include <string>
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

        /// @brief This routine compute one of the following operation on the
        /// distribution function:
        ///  - choice 3 : Fa = Fa * FM0a
        ///  - choice 4 : Fa = Fa / FM0a
        ///
        template <typename ExecutionSpace,
                  typename View6D>
        static inline void
        DoComputeMaxwellianUnshiftedLoc(const Operator& an_operator,
                                        ExecutionSpace  an_execution_space,
                                        uint32_t        choice,
                                        View6D&&        f5d_s) {

            const auto& mu_extent      = an_operator.mu_extent_;
            const auto& vpar_extent    = an_operator.vpar_extent_;
            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
            const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

            const auto& mug    = an_operator.mug_;
            const auto& vparg  = an_operator.vparg_;
            const auto& B_norm = an_operator.B_norm_;
            const auto& n_s    = an_operator.n_s_;
            const auto& T_s    = an_operator.T_s_;

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
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

            switch(choice) {
                case 3: {
                    const auto kernel_choice_3 = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                        const OrdinalType team_index = a_team_handle.league_rank();

                        const OrdinalType itor1     = (team_index) % tor1_extent;
                        const OrdinalType itor2     = (team_index / (tor1_extent)) % tor2_extent;
                        const OrdinalType itor3     = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                        const OrdinalType ispeciesA = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                        const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                        const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                        static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                        const ComputeFloatType Bij    = B_norm(local_domain_r_offset + ir,
                                                               local_domain_theta_offset + itheta);
                        const ComputeFloatType na     = n_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType Ta     = T_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType BijoTa = Bij / Ta;
                        const ComputeFloatType f2oTa  = 0.5 / Ta;
                        const ComputeFloatType coeff  = na / (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta));

                        const koliop::utility::Flatten2 flattened_mu_vpar{mu_extent, vpar_extent};

                        Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle, flattened_mu_vpar.forward()), [&](auto flattened_index) {
                            const auto mu_vpar_loop_body = [&](auto imu, auto ivpar) {
                                const ComputeFloatType v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = Kokkos::max(0.0,
                                                                                                coeff * Kokkos::exp(-v2oTa) * f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA));
                            };
                            flattened_mu_vpar.Call(flattened_index, mu_vpar_loop_body);
                        });
                    };

                    Kokkos::parallel_for("DoComputeMaxwellianUnshiftedLoc3",
                                         a_policy, kernel_choice_3);
                    break;
                }
                case 4: {
                    const auto kernel_choice_4 = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {
                        const OrdinalType team_index = a_team_handle.league_rank();

                        const OrdinalType itor1     = (team_index) % tor1_extent;
                        const OrdinalType itor2     = (team_index / (tor1_extent)) % tor2_extent;
                        const OrdinalType itor3     = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                        const OrdinalType ispeciesA = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                        const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                        const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                        static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                        const ComputeFloatType Bij    = B_norm(local_domain_r_offset + ir,
                                                               local_domain_theta_offset + itheta);
                        const ComputeFloatType na     = n_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType Ta     = T_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType BijoTa = Bij / Ta;
                        const ComputeFloatType f2oTa  = 0.5 / Ta;
                        const ComputeFloatType coeff  = (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta)) / na;

                        const koliop::utility::Flatten2 flattened_mu_vpar{mu_extent, vpar_extent};

                        Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle, flattened_mu_vpar.forward()), [&](auto flattened_index) {
                            const auto mu_vpar_loop_body = [&](auto imu, auto ivpar) {
                                const ComputeFloatType v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = Kokkos::max(0.001,
                                                                                                Kokkos::min(1000.0,
                                                                                                            coeff * Kokkos::exp(v2oTa) * f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA)));
                            };
                            flattened_mu_vpar.Call(flattened_index, mu_vpar_loop_body);
                        });
                    };

                    Kokkos::parallel_for("DoComputeMaxwellianUnshiftedLoc4",
                                         a_policy, kernel_choice_4);
                    break;
                }
                default:
                    throw std::domain_error{"DoComputeMaxwellianUnshiftedLoc: Invalid choice:" + std::to_string(choice)};
                    break;
            }
        }

        void Operator::ComputeMaxwellianUnshiftedLoc(uint32_t choice,
                                                     Flo64*   f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                                       species_extent_);

            {
                DoComputeMaxwellianUnshiftedLoc(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                choice,
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
