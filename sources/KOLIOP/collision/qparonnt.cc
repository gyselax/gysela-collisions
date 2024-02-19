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

        template <typename View6D,
                  typename View4D>
        static inline void
        DoComputeQparOnNT(const Operator& an_operator,
                          const View6D&   f5d_s,
                          View4D&&        q_par_on_nT_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& the_local_domain_r_offset     = an_operator.the_local_domain_r_offset_;
            const auto& the_local_domain_theta_offset = an_operator.the_local_domain_theta_offset_;

            const auto& mug            = an_operator.mug_;
            const auto& vparg          = an_operator.vparg_;
            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& B_norm         = an_operator.B_norm_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;
            const auto& U_par_s        = an_operator.U_par_s_;
            const auto& T_s            = an_operator.T_s_;

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

            const auto the_team_count = the_r_extent * the_theta_extent * the_phi_extent * the_species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            const auto a_policy = PolicyType{
                the_execution_space_pool[Constant::kDefaultExecutionSpacePoolIndex],
                static_cast<int>(the_team_count),
                // NOTE: The thread block size and the Nmu benefit from being
                // close. This is due to the TeamThreadMDRange which
                // parallelizes only the left most index which is Nmu.
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

            const auto the_kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType the_team_index = a_team_handle.league_rank();

                const OrdinalType ir       = (the_team_index) % the_r_extent;
                const OrdinalType itheta   = (the_team_index / (the_r_extent)) % the_theta_extent;
                const OrdinalType iphi     = (the_team_index / (the_r_extent * the_theta_extent)) % the_phi_extent;
                const OrdinalType ispecies = (the_team_index / (the_r_extent * the_theta_extent * the_phi_extent)) /* % the_species_extent */;

                const ComputeFloatType Bij   = B_norm(the_local_domain_r_offset + ir,
                                                      the_local_domain_theta_offset + itheta);
                const ComputeFloatType na    = n_s(ir, itheta, iphi, ispecies);
                const ComputeFloatType Vpara = U_par_s(ir, itheta, iphi, ispecies);
                const ComputeFloatType Ta    = T_s(ir, itheta, iphi, ispecies);
                // TODO(Etienne M): SIMPLIFY as std::pow(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta, -3 / 2);
                const ComputeFloatType coeff = 1.0 / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta));

                ComputeFloatType              q_par_on_nT_s_reduction;
                Kokkos::Sum<ComputeFloatType> the_q_par_on_nT_s_reducer{q_par_on_nT_s_reduction};
                the_q_par_on_nT_s_reducer.init(the_q_par_on_nT_s_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit the_q_par_on_nT_s_reducer followed by team_reduce, a
                // valid semantic.

                Kokkos::parallel_for(
                    // NOTE: The explicit Left iteration direction may not benefit
                    // the CPUs depending on how that is implemented.
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                              std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                the_mu_extent, the_vpar_extent),
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jac = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                     Bstar_s(ivpar, ir, itheta, ispecies);

                        const ComputeFloatType vparg_minus_vpara         = vparg(ivpar) - Vpara;
                        const ComputeFloatType vparg_minus_vpara_squared = vparg_minus_vpara * vparg_minus_vpara;
                        const ComputeFloatType vp2                       = vparg_minus_vpara_squared / (2.0 * Ta);

                        const ComputeFloatType u_tmp   = mug(imu) * Bij / Ta;
                        const ComputeFloatType FMa_tmp = na * Kokkos::exp(-u_tmp - vp2) * coeff;
                        const ComputeFloatType v2      = 2.0 * mug(imu) * Bij + vparg_minus_vpara_squared;

                        the_q_par_on_nT_s_reducer.join(the_q_par_on_nT_s_reducer.reference(),
                                                       0.5 * jac * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * vparg_minus_vpara * v2 *
                                                           (f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) - FMa_tmp));
                    });

                a_team_handle.team_reduce(the_q_par_on_nT_s_reducer);

                // NOTE: This should be a no-op.
                q_par_on_nT_s_reduction = the_q_par_on_nT_s_reducer.reference();

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        q_par_on_nT_s(ir, itheta, iphi, ispecies) = q_par_on_nT_s_reduction / (na * Ta);
                    });
            };

            // NOTE: When this parallel_for is called I observe an unexpected
            // hipEventSynchronize followed by a memcpy and then the
            // hipLaunchKernel.
            // I believe this is due to kokkos copying kernel arguments into the
            // global memory.
            Kokkos::parallel_for("DoComputeQparOnNT",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeQparOnNT(const Flo64* f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<const Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            DoComputeQparOnNT(*this, a_device_visible_f5d_s, q_par_on_nT_s_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
