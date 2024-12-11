// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>
#include <utility>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/tordu.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        /// @brief Correction to keep the local fluid moments constant.
        /// init corresponds to values before collisions.
        /// old corresponds to values after collisions but before correction.
        /// new corresponds to corrected values.
        ///
        template <typename ExecutionSpace,
                  typename View6D,
                  typename View4D0,
                  typename View4D1>
        static inline void
        DoComputeUparTCorrection(const Operator&  an_operator,
                                 ExecutionSpace   an_execution_space,
                                 ComputeFloatType dt,
                                 const View4D0&   input_U_par_s,
                                 View6D&&         f5d_s,
                                 View4D1&&        output_U_par_s) {

            const auto& mu_extent      = an_operator.mu_extent_;
            const auto& vpar_extent    = an_operator.vpar_extent_;
            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
            const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

            const auto& hat_As         = an_operator.hat_As_;
            const auto& mug            = an_operator.mug_;
            const auto& vparg          = an_operator.vparg_;
            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& B_norm         = an_operator.B_norm_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;
            const auto& T_s            = an_operator.T_s_;
            const auto& nu_ab          = an_operator.nu_ab_;
            const auto& q_par_on_nT_s  = an_operator.q_par_on_nT_s_;

            const auto& collision_interspecies = an_operator.collision_interspecies_;

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

            const auto team_count = tor1_extent * tor2_extent * tor3_extent * species_extent;

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

                const OrdinalType itor1     = (team_index) % tor1_extent;
                const OrdinalType itor2     = (team_index / (tor1_extent)) % tor2_extent;
                const OrdinalType itor3     = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                const OrdinalType ispeciesA = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                // NOTE: The explicit Left iteration direction may not benefit
                // the CPUs depending on how that is implemented.
                const auto an_intra_team_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                                            std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                                              mu_extent, vpar_extent);

                static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                ComputeFloatType              impulsion_coll;
                Kokkos::Sum<ComputeFloatType> impulsion_coll_reducer{impulsion_coll};
                impulsion_coll_reducer.init(impulsion_coll_reducer.reference());

                const ComputeFloatType Bij = B_norm(local_domain_r_offset + ir,
                                                    local_domain_theta_offset + itheta);
                const ComputeFloatType na  = n_s(itor1, itor2, itor3, ispeciesA);

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit na_reducer followed by team_reduce, a valid
                // semantic.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = k2Pi *
                                                           Bstar_s(ivpar, ir, itheta, ispeciesA);

                        impulsion_coll_reducer.join(impulsion_coll_reducer.reference(),
                                                    coeff_intdvpar(ivpar) *
                                                        coeff_intdmu(imu) *
                                                        jacob_vel *
                                                        vparg(ivpar) *
                                                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA));
                    });

                a_team_handle.team_reduce(impulsion_coll_reducer);

                // NOTE: This should be a no-op.
                impulsion_coll = impulsion_coll_reducer.reference();

                // Computation of Vpar after collision.
                const ComputeFloatType U_par_coll = impulsion_coll / na;

                ComputeFloatType              T_coll;
                Kokkos::Sum<ComputeFloatType> T_coll_reducer{T_coll};
                T_coll_reducer.init(T_coll_reducer.reference());

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = (k2Pi / na) *
                                                           Bstar_s(ivpar, ir, itheta, ispeciesA);

                        T_coll_reducer.join(T_coll_reducer.reference(),
                                            coeff_intdvpar(ivpar) *
                                                coeff_intdmu(imu) *
                                                (jacob_vel / 3.0) *
                                                ((vparg(ivpar) - U_par_coll) * (vparg(ivpar) - U_par_coll) +
                                                 2.0 * mug(imu) * Bij) *
                                                f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA));
                    });

                a_team_handle.team_reduce(T_coll_reducer);

                // Computation of T after collision.
                // NOTE: This should be a no-op.
                T_coll = T_coll_reducer.reference();

                const ComputeFloatType hat_Aa      = hat_As(ispeciesA);
                const ComputeFloatType Vpara       = input_U_par_s(itor1, itor2, itor3, ispeciesA);
                ComputeFloatType       U_par_new   = Vpara;
                const ComputeFloatType q_par_on_nT = q_par_on_nT_s(itor1, itor2, itor3, ispeciesA);
                const ComputeFloatType T_new       = T_s(itor1, itor2, itor3, ispeciesA);

                // We correct the parallel velocity and temperature.

                for(OrdinalType ispecies = 0; ispecies < species_extent; ++ispecies) {
                    if((ispecies == ispeciesA) || !collision_interspecies) {
                        continue;
                    }

                    const ComputeFloatType hat_Ab         = hat_As(ispecies);
                    const ComputeFloatType Rat            = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const ComputeFloatType Vparb          = input_U_par_s(itor1, itor2, itor3, ispecies);
                    const ComputeFloatType q_parb_on_nbTb = q_par_on_nT_s(itor1, itor2, itor3, ispecies);

                    const ComputeFloatType k_ab = 1.0 / (1.0 +
                                                         hat_Aa * T_s(itor1, itor2, itor3, ispecies) /
                                                             (hat_Ab * T_s(itor1, itor2, itor3, ispeciesA)));
                    const ComputeFloatType k_ba = 1.0 / (1.0 +
                                                         hat_Ab * T_s(itor1, itor2, itor3, ispeciesA) /
                                                             (hat_Aa * T_s(itor1, itor2, itor3, ispecies)));

                    const ComputeFloatType nu = nu_ab(itor1, itor2, itor3, ispeciesA, ispecies);

                    const ComputeFloatType R_ab = -nu * (Vpara -
                                                         Rat * Vparb -
                                                         0.6 * k_ab * q_par_on_nT +
                                                         Rat * 0.6 * k_ba * q_parb_on_nbTb);
                    U_par_new += dt * R_ab;
                }

                const ComputeFloatType coeff_Max_coll = na / (k2Pi * T_coll * Kokkos::sqrt(k2Pi * T_coll));

                const ComputeFloatType coeff_Max_new = na / (k2Pi * T_new * Kokkos::sqrt(k2Pi * T_new));

                // We remove the old Maxwellian and add the new one.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType u_coll = mug(imu) * Bij / T_coll;
                        const ComputeFloatType u_new  = mug(imu) * Bij / T_new;

                        const ComputeFloatType FMa_coll = coeff_Max_coll * Kokkos::exp(-u_coll -
                                                                                       (vparg(ivpar) - U_par_coll) *
                                                                                           (vparg(ivpar) - U_par_coll) / (2.0 * T_coll));
                        const ComputeFloatType FMa_new  = coeff_Max_new * Kokkos::exp(-u_new -
                                                                                      (vparg(ivpar) - U_par_new) *
                                                                                          (vparg(ivpar) - U_par_new) / (2.0 * T_new));

                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) += FMa_new - FMa_coll;
                    });

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        output_U_par_s(itor1, itor2, itor3, ispeciesA) = U_par_new;
                    });
            };

            Kokkos::parallel_for("DoComputeUparTCorrection",
                                 a_policy, kernel);
        }

        void Operator::ComputeUparTCorrection(ComputeFloatType dt,
                                              Flo64*           f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                                       species_extent_);

            {
                // ComputeUparTCorrection reads U_par_s_input_ and outputs to
                // U_par_s_. This is required if more than 1 species is used, we
                // can not write in U_par_s_ while it is still being read (by
                // other threads, species, thread block, etc.).

                // We will not have a race when swapping the buffer just below
                // because when a kernel is launched, and we have the guarantee
                // that the previous are in queue their respective streams and
                // that they carry their state which in this case, would be a
                // copy of U_par_s_ before the swap.
                std::swap(U_par_s_, U_par_s_input_);

                DoComputeUparTCorrection(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                         dt,
                                         U_par_s_input_, a_device_visible_f5d_s, U_par_s_);
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
