// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/select.h"

namespace koliop {
    namespace collision {

        template <typename View6D,
                  typename View4D>
        static inline void
        DoComputeMoments(const Operator& an_operator,
                         View6D&&        f5d_s,
                         View4D&&        n_s,
                         View4D&&        U_par_s,
                         View4D&&        T_s) {

            const auto& the_mu_extent      = an_operator.the_mu_extent_;
            const auto& the_vpar_extent    = an_operator.the_vpar_extent_;
            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_theta_extent   = an_operator.the_theta_extent_;
            const auto& the_phi_extent     = an_operator.the_phi_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& mug            = an_operator.mug_;
            const auto& vparg          = an_operator.vparg_;
            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& B_norm         = an_operator.B_norm_;
            const auto& Bstar_s        = an_operator.Bstar_s_;

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            // TODO(Etienne M): Make sure the accesses are coalesced on GPU
            // and sequential on CPU.

            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {

                const OrdinalType the_team_index = a_team_handle.league_rank();

                const OrdinalType ir       = (the_team_index) % the_r_extent;
                const OrdinalType itheta   = (the_team_index / (the_r_extent)) % the_theta_extent;
                const OrdinalType iphi     = (the_team_index / (the_r_extent * the_theta_extent)) % the_phi_extent;
                const OrdinalType ispecies = (the_team_index / (the_r_extent * the_theta_extent * the_phi_extent)) /* % the_species_extent */;

                const auto an_intra_team_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
                                                                            std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                                              the_mu_extent, the_vpar_extent);

                ComputeFloatType na;
                ComputeFloatType vpara;

                Kokkos::Sum<ComputeFloatType> the_na_reducer{na};
                Kokkos::Sum<ComputeFloatType> the_vpara_reducer{vpara};

                the_na_reducer.init(the_na_reducer.reference());
                the_vpara_reducer.init(the_vpara_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit the_na_reducer followed by team_reduce, a valid
                // semantic.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType f5d_s_value = Kokkos::max(1.0E-40,
                                                                         f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                        const ComputeFloatType ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                                      (coeff_intdmu(imu) * f5d_s_value);

                        the_na_reducer.join(the_na_reducer.reference(),
                                            ftmp);
                        the_vpara_reducer.join(the_vpara_reducer.reference(),
                                               vparg(ivpar) * ftmp);

                        f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) = f5d_s_value;
                    });

                a_team_handle.team_reduce(the_na_reducer);
                a_team_handle.team_reduce(the_vpara_reducer);

                // NOTE: This should be a no-op.
                na = the_na_reducer.reference();

                vpara = the_vpara_reducer.reference() / na;

                ComputeFloatType Ta;

                Kokkos::Sum<ComputeFloatType> the_Ta_reducer{Ta};

                the_Ta_reducer.init(the_Ta_reducer.reference());

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        // TODO(Etienne M): Avoid this recomputation ? On
                        // Gysela fortran, a temporary buffer is used but
                        // that would hardly fit on a GPU.
                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const ComputeFloatType ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                                      (coeff_intdmu(imu) * f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                        the_Ta_reducer.join(the_Ta_reducer.reference(),
                                            ((vparg(ivpar) - vpara) * (vparg(ivpar) - vpara) +
                                             2.0 * mug(imu) * B_norm(ir, itheta)) *
                                                ftmp);
                    });

                a_team_handle.team_reduce(the_Ta_reducer);

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        // NOTE: This should be a no-op.
                        Ta = the_Ta_reducer.reference();

                        Ta = Ta / (3.0 * na);
                        na = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * na;

                        n_s(ir, itheta, iphi, ispecies)     = na;
                        U_par_s(ir, itheta, iphi, ispecies) = vpara;
                        T_s(ir, itheta, iphi, ispecies)     = Ta;
                    });
            };

            const auto the_team_count = the_r_extent * the_theta_extent * the_phi_extent * the_species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            // TODO(Etienne M): TEAM hardcoded size, check for auto tuning
            // opportunities.
            const auto a_policy = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<256, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>{
                the_execution_space_pool[Constant::kDefaultExecutionSpacePoolIndex],
                static_cast<int>(the_team_count),
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, 256)};

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeMoments(Flo64* f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            DoComputeMoments(*this, a_device_visible_f5d_s, n_s_, U_par_s_, T_s_);

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
