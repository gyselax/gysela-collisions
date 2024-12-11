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
#include "KOLIOP/utility/tordu.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        /// @brief Computes the fluid moments:
        ///   na =  int 2 * Pi * Bstar * dmu * dvpar * Fa
        ///   Vpara = (1/na) * int 2 * Pi * Bstar * dmu * dvpar * vpar * Fa
        ///   Ta = (1/(3*na)) * int 2 * Pi * Bstar * dmu * dvpar * Fa * &
        ///        (2 * mu *B + (vpar-Vpara)^2)
        ///
        template <typename ExecutionSpace,
                  typename View6D,
                  typename View4D>
        static inline void
        DoComputeMoments(const Operator& an_operator,
                         ExecutionSpace  an_execution_space,
                         View6D&&        f5d_s,
                         View4D&&        n_s,
                         View4D&&        U_par_s,
                         View4D&&        T_s) {

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
                // NOTE: In the thread blocks, we parallelize on Nmu. If we have
                // more thread than Nmu we may be wasting resources.
                // This is mitigated by the occupancy and the wavefront
                // granularity:
                // - Say you have a kernel using 64 threads while Nmu is 64, that
                // is perfect. If Nmu is now 48, we wast 16 threads but in any
                // case, on AMD GPUs, we can only allocate threads in chunks on
                // 64 called a wavefront.
                // - Also, if we have an kernel with 64 threads and whose
                // occupancy is limited by something other that proportional to
                // the thread amount (LDS ?), it could be beneficial to use more
                // threads say, 128. This would cover the case where Nmu is > 64
                // and due to the occupancy, you would not waste threads (there
                // is a maximum wavefront count per compute unit). That said, on
                // future GPUs where the factor limiting the occupancy mentioned
                // above disappear, one would then start wasting threads.
                // - That said, having less wavefront on a kernel will increase
                // the cache hits. This is significant if you know for a fact
                // that few of your kernels are need to reach peak memory
                // throughput.
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64 /* 128 */, 32)};

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType itor1    = (team_index) % tor1_extent;
                const OrdinalType itor2    = (team_index / (tor1_extent)) % tor2_extent;
                const OrdinalType itor3    = (team_index / (tor1_extent * tor2_extent)) % tor3_extent;
                const OrdinalType ispecies = (team_index / (tor1_extent * tor2_extent * tor3_extent)) /* % species_extent */;

                const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                // NOTE: The explicit Left iteration direction may not benefit
                // the CPUs depending on how that is implemented.
                const auto an_intra_team_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                                                                            std::remove_reference_t<decltype(a_team_handle)>>(a_team_handle,
                                                                                                                              mu_extent, vpar_extent);

                ComputeFloatType na;
                ComputeFloatType vpara;

                // NOTE: As I reported issues with reduction in team policies
                // (mentioned here https://github.com/kokkos/kokkos/pull/6511) I
                // prefer not relying on them while not they are not stable.
                // Thus the explicit reductions which IMO, also reduce the load
                // of implicit stuff you need to know and in the end, make the
                // code more maintainable.

                Kokkos::Sum<ComputeFloatType> na_reducer{na};
                Kokkos::Sum<ComputeFloatType> vpara_reducer{vpara};

                na_reducer.init(na_reducer.reference());
                vpara_reducer.init(vpara_reducer.reference());

                // TODO(Etienne M): Is parallel_for + TeamThreadMDRange +
                // explicit na_reducer followed by team_reduce, a valid
                // semantic.

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        const ComputeFloatType f5d_s_value = Kokkos::max(1.0E-40,
                                                                         f5d_s(imu, ivpar, itor1, itor2, itor3, ispecies));

                        const ComputeFloatType ftmp = Bstar_s(ivpar, ir, itheta, ispecies) *
                                                      coeff_intdmu(imu) *
                                                      coeff_intdvpar(ivpar) *
                                                      f5d_s_value;

                        na_reducer.join(na_reducer.reference(),
                                        ftmp);
                        vpara_reducer.join(vpara_reducer.reference(),
                                           vparg(ivpar) * ftmp);

                        f5d_s(imu, ivpar, itor1, itor2, itor3, ispecies) = f5d_s_value;
                    });

                a_team_handle.team_reduce(na_reducer);
                a_team_handle.team_reduce(vpara_reducer);

                // NOTE: This should be a no-op.
                na    = na_reducer.reference();
                vpara = vpara_reducer.reference();

                vpara = vpara / na;

                ComputeFloatType              Ta;
                Kokkos::Sum<ComputeFloatType> Ta_reducer{Ta};
                Ta_reducer.init(Ta_reducer.reference());

                Kokkos::parallel_for(
                    an_intra_team_policy,
                    [&](auto imu, auto ivpar) {
                        // TODO(Etienne M): Avoid this recomputation ? On
                        // Gysela fortran, a temporary buffer is used but
                        // that would hardly fit on a GPU.
                        // TODO(Etienne M): LOAD Hoist the invariant load.
                        const ComputeFloatType ftmp = Bstar_s(ivpar, ir, itheta, ispecies) *
                                                      coeff_intdmu(imu) *
                                                      coeff_intdvpar(ivpar) *
                                                      f5d_s(imu, ivpar, itor1, itor2, itor3, ispecies);

                        Ta_reducer.join(Ta_reducer.reference(),
                                        (((vparg(ivpar) - vpara) * (vparg(ivpar) - vpara)) +
                                         (2.0 * mug(imu) * B_norm(local_domain_r_offset + ir, local_domain_theta_offset + itheta))) *
                                            ftmp);
                    });

                a_team_handle.team_reduce(Ta_reducer);

                // NOTE: This should be a no-op.
                Ta = Ta_reducer.reference();

                static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        Ta = Ta / (3.0 * na);

                        na = k2Pi * na;

                        n_s(itor1, itor2, itor3, ispecies)     = na;
                        U_par_s(itor1, itor2, itor3, ispecies) = vpara;
                        T_s(itor1, itor2, itor3, ispecies)     = Ta;
                    });
            };

            Kokkos::parallel_for("DoComputeMoments",
                                 a_policy, kernel);
        }

        void Operator::ComputeMoments(Flo64* f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                                       species_extent_);

            DoComputeMoments(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                             a_device_visible_f5d_s, n_s_, U_par_s_, T_s_);

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
