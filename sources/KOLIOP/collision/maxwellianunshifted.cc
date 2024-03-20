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
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        template <typename ExecutionSpace,
                  typename View6D>
        static inline void
        DoComputeMaxwellianUnshiftedLoc(const Operator& an_operator,
                                        ExecutionSpace  an_execution_space,
                                        uint32_t        choice,
                                        OrdinalType     the_species_index,
                                        View6D&&        f5d_s) {

            const auto& the_mu_extent    = an_operator.the_mu_extent_;
            const auto& the_vpar_extent  = an_operator.the_vpar_extent_;
            const auto& the_r_extent     = an_operator.the_r_extent_;
            const auto& the_theta_extent = an_operator.the_theta_extent_;
            const auto& the_phi_extent   = an_operator.the_phi_extent_;

            const auto& the_local_domain_r_offset     = an_operator.the_local_domain_r_offset_;
            const auto& the_local_domain_theta_offset = an_operator.the_local_domain_theta_offset_;

            const auto& mug    = an_operator.mug_;
            const auto& vparg  = an_operator.vparg_;
            const auto& B_norm = an_operator.B_norm_;
            const auto& n_s    = an_operator.n_s_;
            const auto& T_s    = an_operator.T_s_;

            const auto the_team_count = the_r_extent * the_theta_extent * the_phi_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            using PolicyType = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>;

            const auto a_policy = PolicyType{
                an_execution_space,
                static_cast<int>(the_team_count),
                // NOTE: The thread block size and the Nmu benefit from being
                // close. This is due to the TeamThreadMDRange which
                // parallelizes only the left most index which is Nmu.
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

            switch(choice) {
                case 3: {
                    const auto the_kernel_choice_3 = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                        const OrdinalType the_team_index = a_team_handle.league_rank();

                        const OrdinalType ir     = (the_team_index) % the_r_extent;
                        const OrdinalType itheta = (the_team_index / (the_r_extent)) % the_theta_extent;
                        const OrdinalType iphi   = (the_team_index / (the_r_extent * the_theta_extent)) /* % the_phi_extent */;

                        const ComputeFloatType Bij    = B_norm(the_local_domain_r_offset + ir,
                                                               the_local_domain_theta_offset + itheta);
                        const ComputeFloatType na     = n_s(ir, itheta, iphi, the_species_index);
                        const ComputeFloatType Ta     = T_s(ir, itheta, iphi, the_species_index);
                        const ComputeFloatType BijoTa = Bij / Ta;
                        const ComputeFloatType f2oTa  = 0.5 / Ta;
                        const ComputeFloatType coeff  = na / (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta));

                        Kokkos::parallel_for(
                            // NOTE: The explicit Left iteration direction may not benefit
                            // the CPUs depending on how that is implemented.
                            Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                      std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                        the_mu_extent, the_vpar_extent),
                            [&](auto imu, auto ivpar) {
                                const ComputeFloatType v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index) = Kokkos::max(0.0,
                                                                                               coeff * Kokkos::exp(-v2oTa) * f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index));
                            });
                    };

                    Kokkos::parallel_for("DoComputeMaxwellianUnshiftedLoc3",
                                         a_policy, the_kernel_choice_3);
                    break;
                }
                case 4: {
                    const auto the_kernel_choice_4 = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {
                        const OrdinalType the_team_index = a_team_handle.league_rank();

                        const OrdinalType ir     = (the_team_index) % the_r_extent;
                        const OrdinalType itheta = (the_team_index / (the_r_extent)) % the_theta_extent;
                        const OrdinalType iphi   = (the_team_index / (the_r_extent * the_theta_extent)) /* % the_phi_extent */;

                        const ComputeFloatType Bij    = B_norm(the_local_domain_r_offset + ir,
                                                               the_local_domain_theta_offset + itheta);
                        const ComputeFloatType na     = n_s(ir, itheta, iphi, the_species_index);
                        const ComputeFloatType Ta     = T_s(ir, itheta, iphi, the_species_index);
                        const ComputeFloatType BijoTa = Bij / Ta;
                        const ComputeFloatType f2oTa  = 0.5 / Ta;
                        const ComputeFloatType coeff  = (koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta * Kokkos::sqrt(koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Ta)) / na;

                        Kokkos::parallel_for(
                            // NOTE: The explicit Left iteration direction may not benefit
                            // the CPUs depending on how that is implemented.
                            Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                      std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                        the_mu_extent, the_vpar_extent),
                            [&](auto imu, auto ivpar) {
                                const ComputeFloatType v2oTa = mug(imu) * BijoTa + vparg(ivpar) * vparg(ivpar) * f2oTa;

                                f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index) = Kokkos::max(0.001,
                                                                                               Kokkos::min(1000.0,
                                                                                                           coeff * Kokkos::exp(v2oTa) * f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index)));
                            });
                    };

                    Kokkos::parallel_for("DoComputeMaxwellianUnshiftedLoc4",
                                         a_policy, the_kernel_choice_4);
                    break;
                }
                default:
                    throw std::domain_error{"DoComputeMaxwellianUnshiftedLoc: Invalid choice:" + std::to_string(choice)};
                    break;
            }
        }

        void Operator::ComputeMaxwellianUnshiftedLoc(uint32_t    choice,
                                                     Flo64*      f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            for(OrdinalType the_species_index = 0; the_species_index < the_species_extent_; ++the_species_index) {
                DoComputeMaxwellianUnshiftedLoc(*this, the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                choice, the_species_index,
                                                a_device_visible_f5d_s);
            }

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
