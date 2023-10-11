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

        template <typename View6D>
        static inline void
        DoComputeDensityCorrection(const Operator& an_operator,
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

            const auto& the_execution_space_pool = an_operator.the_execution_space_pool_;

            const auto& pSpecies_id = an_operator.pSpecies_id_;

            // TODO(Etienne M): Make sure the accesses are coalesced on GPU
            // and sequential on CPU.

            const auto the_kernel = KOKKOS_LAMBDA(const auto& a_team_handle) {

                const OrdinalType the_team_index = a_team_handle.league_rank();

                const OrdinalType ir     = (the_team_index) % the_r_extent;
                const OrdinalType itheta = (the_team_index / (the_r_extent)) % the_theta_extent;
                const OrdinalType iphi   = (the_team_index / (the_r_extent * the_theta_extent)) /* % the_phi_extent */;

                const auto an_intra_team_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
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
                                                           Bstar_s(ivpar, ir, itheta, pSpecies_id);
                        const ComputeFloatType ftmp = Kokkos::max(f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id),
                                                                  0.0);

                        the_na_reducer.join(the_na_reducer.reference(),
                                            jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp);

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = ftmp;
                    });

                a_team_handle.team_reduce(the_na_reducer);

                // NOTE: This should be a no-op.
                na = the_na_reducer.reference();

                const ComputeFloatType na_inverse = 1.0 / na;

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) *= n_s(ir, itheta, iphi, pSpecies_id) * na_inverse;
                    });
            };

            const auto the_team_count = the_r_extent * the_theta_extent * the_phi_extent;

            // Due to the inherent Cuda/HIP thread block limit.
            KOLIOP_ASSERT(the_team_count <= std::numeric_limits<int>::max());

            // TODO(Etienne M): TEAM hardcoded size, check for auto tuning
            // opportunities.
            const auto a_policy = Kokkos::TeamPolicy<
                // Kokkos::LaunchBounds<512, 1>, Kokkos::IndexType<OrdinalType>,
                Constant::DeviceExecutionSpaceType>{
                the_execution_space_pool[Constant::kDefaultExecutionSpacePoolIndex],
                static_cast<int>(the_team_count),
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO)};

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

            DoComputeDensityCorrection(*this, a_device_visible_f5d_s);

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
