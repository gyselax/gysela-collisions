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

        template <typename ExecutionSpace,
                  typename View6D>
        static inline void
        DoComputeDensityCorrection(const Operator& an_operator,
                                   ExecutionSpace  an_execution_space,
                                   OrdinalType     the_species_index,
                                   View6D&&        f5d_s) {

            const auto& the_mu_extent    = an_operator.the_mu_extent_;
            const auto& the_vpar_extent  = an_operator.the_vpar_extent_;
            const auto& the_r_extent     = an_operator.the_r_extent_;
            const auto& the_theta_extent = an_operator.the_theta_extent_;
            const auto& the_phi_extent   = an_operator.the_phi_extent_;

            const auto& coeff_intdmu   = an_operator.coeff_intdmu_;
            const auto& coeff_intdvpar = an_operator.coeff_intdvpar_;
            const auto& Bstar_s        = an_operator.Bstar_s_;
            const auto& n_s            = an_operator.n_s_;

            // TODO(Etienne M): Make sure the accesses are sequential on CPU.

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

                ComputeFloatType              na;
                Kokkos::Sum<ComputeFloatType> the_na_reducer{na};
                the_na_reducer.init(the_na_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit the_na_reducer followed by team_reduce, a valid
                // semantic.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType jacob_vel = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi *
                                                           Bstar_s(ivpar, ir, itheta, the_species_index);
                        const ComputeFloatType ftmp = Kokkos::max(f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index),
                                                                  0.0);

                        the_na_reducer.join(the_na_reducer.reference(),
                                            jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp);

                        f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index) = ftmp;
                    });

                a_team_handle.team_reduce(the_na_reducer);

                // NOTE: This should be a no-op.
                na = the_na_reducer.reference();

                const ComputeFloatType na_inverse = 1.0 / na;

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        f5d_s(imu, ivpar, ir, itheta, iphi, the_species_index) *= n_s(ir, itheta, iphi, the_species_index) * na_inverse;
                    });
            };

            Kokkos::parallel_for("DoComputeDensityCorrection",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeDensityCorrection(Flo64* f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       the_mu_extent_, the_vpar_extent_,
                                                                                                                       the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                                       the_species_extent_);

            for(OrdinalType the_species_index = 0; the_species_index < the_species_extent_; ++the_species_index) {
                DoComputeDensityCorrection(*this, the_execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                           the_species_index,
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
