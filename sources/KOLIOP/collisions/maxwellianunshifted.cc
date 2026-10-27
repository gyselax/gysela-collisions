// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <stdexcept>
#include <string>
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

            const auto& mug    = an_operator.mug_;
            const auto& vparg  = an_operator.vparg_;
            const auto& B_norm = an_operator.B_norm_;
            const auto& n_s    = an_operator.n_s_;
            const auto& T_s    = an_operator.T_s_;

            // NOTE: We use scratch memory to avoid overwriting the T_s.
            using ScratchMemory = Kokkos::View<ComputeFloatType*,
                                               Constant::DeviceExecutionSpaceType::scratch_memory_space,
                                               // NOTE: Be careful with the restrict.
                                               Kokkos::MemoryTraits<Kokkos::Restrict>>;

            const uint64_t scratch_memory_bytes_per_team = (mu_extent) * sizeof(ComputeFloatType);

            const auto team_count = tor1_extent * tor2_extent * tor3_extent * species_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            auto a_policy = PolicyType{
                an_execution_space,
                static_cast<int>(team_count),
                // NOTE: See the note in the DoComputeMoments kernel
                // regarding thread block size.
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 256, 128)};

            a_policy.set_scratch_size(0,
                                      Kokkos::PerTeam(ScratchMemory::shmem_size(scratch_memory_bytes_per_team)));

            switch(choice) {
                case 3: {
                    const auto kernel_choice_3 = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                        const OrdinalType team_index = a_team_handle.league_rank();

                        const OrdinalType itor1     = (team_index) % tor1_extent;
                        const OrdinalType itor2     = (team_index / (tor1_extent)) % tor2_extent;
                        const OrdinalType itor3     = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                        const OrdinalType ispeciesA = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                        // TODO(Etienne M): Should we use a_team_handle.team_barrier() ?
                        ScratchMemory parameter0{a_team_handle.team_scratch(0),
                                                 static_cast<std::size_t>(mu_extent)};

                        static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                        const ComputeFloatType Bij    = B_norm(itor2, itor1, itor3);
                        const ComputeFloatType na     = n_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType Ta     = T_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType BijoTa = Bij / Ta;
                        const ComputeFloatType f2oTa  = 0.5 / Ta;
                        const ComputeFloatType coeff0 = na / (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta));

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(a_team_handle,
                                                    mu_extent),
                            [&](auto imu) {
                                parameter0(imu) = coeff0 * Kokkos::exp(-(mug(imu) * BijoTa));
                            });

                        a_team_handle.team_barrier();

                        using koliop::utility::IntraTeam;
                        IntraTeam::Iterate2D(
                            mu_extent, vpar_extent,
                            a_team_handle, [&](auto imu, auto ivpar) {
                                // const ComputeFloatType v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                // f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = Kokkos::max(0.0,
                                //                                                                 coeff0 * Kokkos::exp(-v2oTa) * f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA));

                                const ComputeFloatType variable0 = parameter0(imu) *
                                                                   Kokkos::exp(-(vparg(ivpar) * vparg(ivpar) * f2oTa));

                                f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = Kokkos::max(0.0,
                                                                                                variable0 * f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA));
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

                        // TODO(Etienne M): Should we use a_team_handle.team_barrier() ?
                        ScratchMemory parameter0{a_team_handle.team_scratch(0),
                                                 static_cast<std::size_t>(mu_extent)};

                        static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                        const ComputeFloatType Bij    = B_norm(itor2, itor1, itor3);
                        const ComputeFloatType na     = n_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType Ta     = T_s(itor1, itor2, itor3, ispeciesA);
                        const ComputeFloatType BijoTa = Bij / Ta;
                        const ComputeFloatType f2oTa  = 0.5 / Ta;
                        const ComputeFloatType coeff0 = (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta)) / na;

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(a_team_handle,
                                                    mu_extent),
                            [&](auto imu) {
                                parameter0(imu) = coeff0 * Kokkos::exp(mug(imu) * BijoTa);
                            });

                        a_team_handle.team_barrier();

                        using koliop::utility::IntraTeam;
                        IntraTeam::Iterate2D(
                            mu_extent, vpar_extent,
                            a_team_handle, [&](auto imu, auto ivpar) {
                                // const ComputeFloatType v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                // f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = Kokkos::max(0.001,
                                //                                                                 Kokkos::min(1000.0,
                                //                                                                             coeff0 * Kokkos::exp(v2oTa) * f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA)));

                                const ComputeFloatType variable0 = parameter0(imu) *
                                                                   Kokkos::exp(vparg(ivpar) * vparg(ivpar) * f2oTa);

                                f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = Kokkos::max(0.001,
                                                                                                Kokkos::min(1000.0,
                                                                                                            variable0 * f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA)));
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

        void Operator::ComputeMaxwellianUnshiftedLoc(uint32_t                                  choice,
                                                     const koliop::utility::View::UDL6<Flo64>& f5d_s) const {
            KOLIOP_COLLISIONS_CHECK_DISTRIBUTION_EXTENTS(f5d_s);

            if(!(choice == 3 || choice == 4)) {
                throw std::runtime_error("KOLIOP_STATUS_INVALID_VALUE");
            }
            DoComputeMaxwellianUnshiftedLoc(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                            choice,
                                            f5d_s);
        }

    } // namespace collisions
} // namespace koliop
