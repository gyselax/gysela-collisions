// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/iterate.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/tordu.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        /// @brief Computation of q_para/(na*Ta):
        ///   q_par_on_nT = 0.5*int (2 * Pi * Bstar) *  dmu * dvpar * &
        ///       (vpar-Vpara) * ((vpar-Vpara)^2 + vperp^2) * (Fa - FMa)
        ///  Indeed  q_par_on_nT = 0 for a Maxwellian distribution function
        ///
        template <typename ExecutionSpace,
                  typename View6D,
                  typename View4D>
        static inline void
        DoComputeQparOnNT(const Operator& an_operator,
                          ExecutionSpace  an_execution_space,
                          const View6D&   f5d_s,
                          View4D&&        q_par_on_nT_s) {

            const auto& mu_extent      = an_operator.mu_extent_;
            const auto& vpar_extent    = an_operator.vpar_extent_;
            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
            const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

            const auto& mug            = an_operator.mug_;
            const auto& vparg          = an_operator.vparg_;
            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& B_norm         = an_operator.B_norm_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;
            const auto& U_par_s        = an_operator.U_par_s_;
            const auto& T_s            = an_operator.T_s_;

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
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 128, 32)};

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType itor1    = (team_index) % tor1_extent;
                const OrdinalType itor2    = (team_index / (tor1_extent)) % tor2_extent;
                const OrdinalType itor3    = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                const OrdinalType ispecies = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                const ComputeFloatType Bij   = B_norm(local_domain_r_offset + ir,
                                                      local_domain_theta_offset + itheta);
                const ComputeFloatType na    = n_s(itor1, itor2, itor3, ispecies);
                const ComputeFloatType Vpara = U_par_s(itor1, itor2, itor3, ispecies);
                const ComputeFloatType Ta    = T_s(itor1, itor2, itor3, ispecies);
                // TODO(Etienne M): SIMPLIFY as std::pow(k2Pi * Ta, -3 / 2);
                const ComputeFloatType coeff = 1.0 / (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta));

                ComputeFloatType              q_par_on_nT_s_reduction;
                Kokkos::Sum<ComputeFloatType> q_par_on_nT_s_reducer{q_par_on_nT_s_reduction};
                q_par_on_nT_s_reducer.init(q_par_on_nT_s_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadRange +
                // explicit q_par_on_nT_s_reducer followed by team_reduce, a
                // valid semantic.

                using koliop::utility::IntraTeam;
                IntraTeam::Iterate2D(
                    mu_extent, vpar_extent,
                    a_team_handle, [&](auto imu, auto ivpar) {
                        // NOTE: This is not vectorized on CPU due to the
                        // reduction.

                        const ComputeFloatType jac = k2Pi *
                                                     Bstar_s(ivpar, ir, itheta, ispecies);

                        const ComputeFloatType vparg_minus_vpara         = vparg(ivpar) - Vpara;
                        const ComputeFloatType vparg_minus_vpara_squared = vparg_minus_vpara * vparg_minus_vpara;
                        const ComputeFloatType vp2                       = vparg_minus_vpara_squared / (2.0 * Ta);

                        const ComputeFloatType u_tmp   = mug(imu) * Bij / Ta;
                        const ComputeFloatType FMa_tmp = na * Kokkos::exp(-u_tmp - vp2) * coeff;
                        const ComputeFloatType v2      = 2.0 * mug(imu) * Bij + vparg_minus_vpara_squared;

                        q_par_on_nT_s_reducer.join(q_par_on_nT_s_reducer.reference(),
                                                   0.5 * jac * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * vparg_minus_vpara * v2 *
                                                       (f5d_s(imu, ivpar, itor1, itor2, itor3, ispecies) - FMa_tmp));
                    });

                a_team_handle.team_reduce(q_par_on_nT_s_reducer);

                // NOTE: This should be a no-op.
                q_par_on_nT_s_reduction = q_par_on_nT_s_reducer.reference();

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        q_par_on_nT_s(itor1, itor2, itor3, ispecies) = q_par_on_nT_s_reduction / (na * Ta);
                    });
            };

            // NOTE: When this parallel_for is called I observe an unexpected
            // hipEventSynchronize followed by a memcpy and then the
            // hipLaunchKernel.
            // I believe this is due to kokkos copying kernel arguments into the
            // global memory.
            Kokkos::parallel_for("DoComputeQparOnNT",
                                 a_policy, kernel);
        }

        void Operator::ComputeQparOnNT(const Flo64* f5d_s) {

            // auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<const Flo64>,
            //                                                                    koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
            //                                                                                                            f5d_s, "f5d_s",
            //                                                                                                            mu_extent_, vpar_extent_,
            //                                                                                                            tor1_extent_, tor2_extent_, tor3_extent_,
            //                                                                                                            species_extent_);

            DoComputeQparOnNT(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                              a_device_visible_f5d_s_, q_par_on_nT_s_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
