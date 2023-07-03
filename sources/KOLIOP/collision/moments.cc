// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

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

            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {

                const auto the_team_index = a_team_handle.league_rank();

                const auto ir       = (the_team_index) % the_r_extent;
                const auto itheta   = (the_team_index / (the_r_extent)) % the_theta_extent;
                const auto iphi     = (the_team_index / (the_r_extent * the_r_extent)) % the_phi_extent;
                const auto ispecies = (the_team_index / (the_r_extent * the_r_extent * the_phi_extent)) /* % the_species_extent */;

                // TODO(Etienne M): Make sure the accesses are coalesced.
                // TODO(Etienne M): Make sure an TeamMDRange followed by
                // a_team_handle.reduce is not faster than a parallel_reduce of
                // TeamThreadRange.

                ComputeFloatType na;
                ComputeFloatType vpara;

                const uint32_t the_flattened_extent = the_mu_extent * the_vpar_extent;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(a_team_handle,
                                            the_flattened_extent),
                    [&](const auto&       the_flattened_index,
                        ComputeFloatType& na_reduction,
                        ComputeFloatType& vpara_reduction) {
                        const auto imu   = (the_flattened_index) % the_mu_extent;
                        const auto ivpar = (the_flattened_index / (the_mu_extent)) /* % the_vpar_extent */;

                        const ComputeFloatType f5d_s_value = Kokkos::max(1.0E-40,
                                                                         f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));

                        const ComputeFloatType ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                                      (coeff_intdmu(imu) * f5d_s_value);

                        na_reduction += ftmp;
                        vpara_reduction += vparg(ivpar) * ftmp;

                        f5d_s(imu, ivpar, ir, itheta, iphi, ispecies) = f5d_s_value;
                    },
                    Kokkos::Sum<ComputeFloatType>{na},
                    Kokkos::Sum<ComputeFloatType>{vpara});

                // NOTE: No implicit barrier the end of the reduction due to the
                // explicit reducer.

                a_team_handle.team_barrier(); // Memory fence.

                vpara = vpara / na;

                ComputeFloatType Ta;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(a_team_handle,
                                            the_flattened_extent),
                    [&](const auto&       the_flattened_index,
                        ComputeFloatType& Ta_reduction) {
                        const auto imu   = (the_flattened_index) % the_mu_extent;
                        const auto ivpar = (the_flattened_index / (the_mu_extent)) /* % the_vpar_extent */;

                        // TODO(Etienne M): Avoid this recomputation ? On
                        // Gysela fortran, a temporary buffer is used but
                        // that would hardly fit on a GPU.
                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const ComputeFloatType ftmp = (Bstar_s(ivpar, ir, itheta, ispecies) * coeff_intdvpar(ivpar)) *
                                                      (coeff_intdmu(imu) * f5d_s(imu, ivpar, ir, itheta, iphi, ispecies));
                        Ta_reduction += ((vparg(ivpar) - vpara) * (vparg(ivpar) - vpara) +
                                         2.0 * mug(imu) * B_norm(ir, itheta)) *
                                        ftmp;
                    },
                    Kokkos::Sum<ComputeFloatType>{Ta});

                // TODO(Etienne M): Check if it is faster using an explicit
                // barrier or use a scalar reducer and benefit from the implicit
                // barrier.
                a_team_handle.team_barrier(); // Memory fence.

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        Ta = Ta / (3.0 * na);
                        na = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * na;

                        n_s(ir, itheta, iphi, ispecies)     = na;
                        U_par_s(ir, itheta, iphi, ispecies) = vpara;
                        T_s(ir, itheta, iphi, ispecies)     = Ta;
                    });
            };

            const uint64_t the_team_count = the_r_extent * the_theta_extent * the_phi_extent * the_species_extent;

            // Due to the inherent cuda/HIP thread block limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            const auto a_policy = Kokkos::TeamPolicy<Constant::DeviceExecutionSpaceType>{static_cast<int>(the_team_count),
                                                                                         Kokkos::AUTO};

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeMoments(Flo64* f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            DoComputeMoments(*this, a_device_visible_f5d_s, n_s_, U_par_s_, T_s_);

            koliop::utility::Copy::ViewToRawData<koliop::utility::View::InternalInputL6<Flo64>,
                                                 koliop::utility::View::ExternalInputL6<Flo64>>(a_device_visible_f5d_s,
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
