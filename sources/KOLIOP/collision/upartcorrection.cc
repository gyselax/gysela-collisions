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
        DoComputeUparTCorrection(const Operator&  an_operator,
                                 ComputeFloatType dt,
                                 View6D&&         f5d_s,
                                 View4D&&         U_par_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& the_local_domain_r_offset     = an_operator.the_local_domain_r_offset_;
            const auto& the_local_domain_theta_offset = an_operator.the_local_domain_theta_offset_;

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

            const auto& pSpecies_id            = an_operator.pSpecies_id_;
            const auto& collision_interspecies = an_operator.collision_interspecies_;

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

            const auto the_team_count = the_r_extent * the_theta_extent * the_phi_extent;

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
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, 64 /* 128 */)};

            const auto the_kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType the_team_index = a_team_handle.league_rank();

                const OrdinalType ir     = (the_team_index) % the_r_extent;
                const OrdinalType itheta = (the_team_index / (the_r_extent)) % the_theta_extent;
                const OrdinalType iphi   = (the_team_index / (the_r_extent * the_theta_extent)) /* % the_phi_extent */;

                // NOTE: The explicit Left iteration direction may not benefit
                // the CPUs depending on how that is implemented.
                const auto an_intra_team_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                                            std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                                              the_mu_extent, the_vpar_extent);

                ComputeFloatType              impulsion_coll;
                Kokkos::Sum<ComputeFloatType> the_impulsion_coll_reducer{impulsion_coll};
                the_impulsion_coll_reducer.init(the_impulsion_coll_reducer.reference());

                const ComputeFloatType Bij = B_norm(the_local_domain_r_offset + ir,
                                                    the_local_domain_theta_offset + itheta);
                const ComputeFloatType na  = n_s(ir, itheta, iphi, pSpecies_id);

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit the_na_reducer followed by team_reduce, a valid
                // semantic.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                           Bstar_s(ivpar, ir, itheta, pSpecies_id);

                        the_impulsion_coll_reducer.join(the_impulsion_coll_reducer.reference(),
                                                        coeff_intdvpar(ivpar) *
                                                            coeff_intdmu(imu) *
                                                            jacob_vel *
                                                            vparg(ivpar) *
                                                            f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id));
                    });

                a_team_handle.team_reduce(the_impulsion_coll_reducer);

                // NOTE: This should be a no-op.
                impulsion_coll = the_impulsion_coll_reducer.reference();

                const ComputeFloatType U_par_coll = impulsion_coll / na;

                ComputeFloatType              T_coll;
                Kokkos::Sum<ComputeFloatType> the_T_coll_reducer{T_coll};
                the_T_coll_reducer.init(the_T_coll_reducer.reference());

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi / na) *
                                                           Bstar_s(ivpar, ir, itheta, pSpecies_id);

                        the_T_coll_reducer.join(the_T_coll_reducer.reference(),
                                                coeff_intdvpar(ivpar) *
                                                    coeff_intdmu(imu) *
                                                    (jacob_vel / 3.0) *
                                                    ((vparg(ivpar) - U_par_coll) * (vparg(ivpar) - U_par_coll) +
                                                     2.0 * mug(imu) * Bij) *
                                                    f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id));
                    });

                a_team_handle.team_reduce(the_T_coll_reducer);

                // NOTE: This should be a no-op.
                T_coll = the_T_coll_reducer.reference();

                const ComputeFloatType hat_Aa      = hat_As(pSpecies_id);
                const ComputeFloatType Vpara       = U_par_s(ir, itheta, iphi, pSpecies_id);
                ComputeFloatType       U_par_new   = Vpara;
                const ComputeFloatType q_par_on_nT = q_par_on_nT_s(ir, itheta, iphi, pSpecies_id);
                const ComputeFloatType T_new       = T_s(ir, itheta, iphi, pSpecies_id);

                for(OrdinalType ispecies = 0; ispecies < the_species_extent; ++ispecies) {
                    if((ispecies == pSpecies_id) || !collision_interspecies) {
                        continue;
                    }

                    const ComputeFloatType hat_Ab         = hat_As(ispecies);
                    const ComputeFloatType Rat            = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const ComputeFloatType Vparb          = U_par_s(ir, itheta, iphi, ispecies);
                    const ComputeFloatType q_parb_on_nbTb = q_par_on_nT_s(ir, itheta, iphi, ispecies);

                    const ComputeFloatType k_ab = 1.0 / (1.0 +
                                                         hat_Aa * T_s(ir, itheta, iphi, ispecies) /
                                                             (hat_Ab * T_s(ir, itheta, iphi, pSpecies_id)));
                    const ComputeFloatType k_ba = 1.0 / (1.0 +
                                                         hat_Ab * T_s(ir, itheta, iphi, pSpecies_id) /
                                                             (hat_Aa * T_s(ir, itheta, iphi, ispecies)));

                    const ComputeFloatType nu = nu_ab(ir, itheta, iphi, pSpecies_id, ispecies);

                    const ComputeFloatType R_ab = -nu * (Vpara -
                                                         Rat * Vparb -
                                                         0.6 * k_ab * q_par_on_nT +
                                                         Rat * 0.6 * k_ba * q_parb_on_nbTb);
                    U_par_new += dt * R_ab;
                }

                const ComputeFloatType coeff_Max_coll = na / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                              T_coll *
                                                              Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * T_coll));

                const ComputeFloatType coeff_Max_new = na / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                             T_new *
                                                             Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * T_new));

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType u_coll = mug(imu) * Bij / T_coll;
                        const ComputeFloatType u_new  = mug(imu) * Bij / T_new;

                        const ComputeFloatType FMa_coll = coeff_Max_coll * Kokkos::exp(-u_coll -
                                                                                       (vparg(ivpar) - U_par_coll) *
                                                                                           (vparg(ivpar) - U_par_coll) / (2 * T_coll));
                        const ComputeFloatType FMa_new  = coeff_Max_new * Kokkos::exp(-u_new -
                                                                                      (vparg(ivpar) - U_par_new) *
                                                                                          (vparg(ivpar) - U_par_new) / (2 * T_new));

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) += FMa_new - FMa_coll;
                    });

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        // NOTE: no need to barrier, we know that all the
                        // threads have a valid U_par_new. Putting this code
                        // before the parallel for requires an additional fence.
                        U_par_s(ir, itheta, iphi, pSpecies_id) = U_par_new;
                    });
            };

            Kokkos::parallel_for("DoComputeUparTCorrection",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeUparTCorrection(ComputeFloatType dt,
                                              Flo64*           f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            DoComputeUparTCorrection(*this, dt, a_device_visible_f5d_s, U_par_s_);

            koliop::utility::Copy::ViewToRawData<koliop::utility::View::InternalInputL6<Flo64>,
                                                 koliop::utility::View::ExternalInputL6<Flo64>>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                a_device_visible_f5d_s,
                                                                                                f5d_s,
                                                                                                the_mu_extent_, the_vpar_extent_,
                                                                                                the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                the_species_extent_);

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
