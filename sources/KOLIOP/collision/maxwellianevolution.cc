// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/iterate.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/tordu.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        /// @brief Make the Maxwellian evolve:
        ///     - remove the Maxwellian to the distribution function Fa = Fa - FMa;
        ///     - evolve temperature;
        ///     - compute the new Maxwellian;
        ///     - add it to the function distribution Fa = Fa + FMa.
        /// Evolution of the temperatures via the term C0:
        ///   Ta = Ta + deltat*sum_b nu_ab * (2*hat_Aa/(hat_Aa + hat_Ab)*(Tb -Ta) +
        ///     (2/3) * Vpara *(Vpara - Vparb))
        ///
        template <typename ExecutionSpace,
                  typename View6D,
                  typename View4D>
        static inline void
        DoComputeMaxwellianEvolution(const Operator&  an_operator,
                                     ExecutionSpace   an_execution_space,
                                     ComputeFloatType dt,
                                     View6D&&         f5d_s,
                                     View4D&&         T_s) {

            const auto& mu_extent      = an_operator.mu_extent_;
            const auto& vpar_extent    = an_operator.vpar_extent_;
            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
            const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

            const auto& hat_As  = an_operator.hat_As_;
            const auto& mug     = an_operator.mug_;
            const auto& vparg   = an_operator.vparg_;
            const auto& B_norm  = an_operator.B_norm_;
            const auto& n_s     = an_operator.n_s_;
            const auto& U_par_s = an_operator.U_par_s_;
            const auto& nu_ab   = an_operator.nu_ab_;

            const ComputeFloatType half_dt = dt / 2.0;

            // NOTE: We use scratch memory to avoid overwriting the T_s.
            using ScratchMemory = Kokkos::View<ComputeFloatType*,
                                               Constant::DeviceExecutionSpaceType::scratch_memory_space,
                                               // NOTE: Be careful with the restrict.
                                               Kokkos::MemoryTraits<Kokkos::Restrict>>;

            const uint64_t scratch_memory_bytes_per_thread = species_extent * sizeof(ComputeFloatType);

            const auto team_count = tor1_extent * tor2_extent * tor3_extent;

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
                koliop::utility::Select<Constant::DeviceExecutionSpaceType>(Kokkos::AUTO, Kokkos::AUTO, 64, 32)};

            a_policy.set_scratch_size(0,
                                      Kokkos::PerThread(ScratchMemory::shmem_size(scratch_memory_bytes_per_thread)));

            const auto kernel = KOKKOS_LAMBDA(const typename PolicyType::member_type& a_team_handle) {

                const OrdinalType team_index = a_team_handle.league_rank();

                const OrdinalType itor1 = (team_index) % tor1_extent;
                const OrdinalType itor2 = (team_index / (tor1_extent)) % tor2_extent;
                const OrdinalType itor3 = (team_index / (tor1_extent * tor2_extent)) /* % tor3_extent */;

                const OrdinalType ir     = koliop::utility::GetRValue(itor1, itor2, itor3);
                const OrdinalType itheta = koliop::utility::GetThetaValue(itor1, itor2, itor3);

                // TODO(Etienne M): Should we use a_team_handle.team_barrier() ?
                ScratchMemory Ta_tmp{a_team_handle.thread_scratch(0),
                                     scratch_memory_bytes_per_thread};

                // NOTE: Iterating on species leads to large strides in
                // memory access. These *bad* accesses are few and proportional
                // to species_extent.

                for(OrdinalType ispeciesA = 0; ispeciesA < species_extent; ++ispeciesA) {
                    const ComputeFloatType hat_Aa = hat_As(ispeciesA);
                    const ComputeFloatType Vpara  = U_par_s(itor1, itor2, itor3, ispeciesA);
                    const ComputeFloatType Ta     = T_s(itor1, itor2, itor3, ispeciesA);

                    ComputeFloatType Ta_new = Ta;

                    // TODO(Etienne M): Split into two loops to avoid the if ?

                    for(OrdinalType ispeciesB = 0; ispeciesB < species_extent; ++ispeciesB) {
                        if(ispeciesA == ispeciesB) {
                            continue;
                        }

                        const ComputeFloatType hat_Ab = hat_As(ispeciesB);
                        const ComputeFloatType Vparb  = U_par_s(itor1, itor2, itor3, ispeciesB);
                        const ComputeFloatType Tb     = T_s(itor1, itor2, itor3, ispeciesB);

                        const ComputeFloatType nu = nu_ab(itor1, itor2, itor3, ispeciesA, ispeciesB);

                        Ta_new += half_dt * nu * (2.0 * hat_Aa * (Tb - Ta) / (hat_Aa + hat_Ab) + (2.0 / 3.0) * Vpara * (Vpara - Kokkos::sqrt(hat_Aa / hat_Ab) * Vparb));
                    }

                    // We can not overwrite the T_s until all the new
                    // T_s are computed.
                    Ta_tmp(ispeciesA) = Ta_new;

                    const ComputeFloatType na = n_s(itor1, itor2, itor3, ispeciesA);

                    static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                    const ComputeFloatType coeff_old = na / (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta));
                    const ComputeFloatType coeff_new = na / (k2Pi * Ta_new * Kokkos::sqrt(k2Pi * Ta_new));

                    const ComputeFloatType Bij = B_norm(local_domain_r_offset + ir,
                                                        local_domain_theta_offset + itheta);

                    const ComputeFloatType bot_old = Bij / Ta;
                    const ComputeFloatType bot_new = Bij / Ta_new;

                    using koliop::utility::IntraTeam;
                    IntraTeam::Iterate2D(
                        mu_extent, vpar_extent,
                        a_team_handle, [&](auto imu, auto ivpar) {
                            const ComputeFloatType vp2_old = (vparg(ivpar) - Vpara) * (vparg(ivpar) - Vpara) / (2.0 * Ta);
                            const ComputeFloatType vp2_new = (vparg(ivpar) - Vpara) * (vparg(ivpar) - Vpara) / (2.0 * Ta_new);

                            // NOTE: on large values of mu, we have a very
                            // "thin" density often close to
                            // 1.000000000000000083E-30. Coupled to values like:
                            // 6.312302637865660744E-11 and
                            // 6.312302638024095558E-11
                            // On double, 1.000000000000000083E-30 - 6.312302637865660744E-11 + 6.312302638024095558E-11 = 1.584348141071684242e-21
                            // but the true values is 1.5843481409999999998 E-21
                            // which represent a relative error of
                            // 4.524525913525214280 E-11. We lost 4-5
                            // significant digits. When compared to the Fortran
                            // code we observe absolute error of 10^-25 and
                            // relative error of 10^-4 !
                            f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) = f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) -
                                                                                coeff_old * Kokkos::exp(-bot_old * mug(imu) - vp2_old) +
                                                                                coeff_new * Kokkos::exp(-bot_new * mug(imu) - vp2_new);
                        });
                }

                Kokkos::single(
                    Kokkos::PerTeam(a_team_handle),
                    [&]() {
                        for(OrdinalType ispeciesA = 0; ispeciesA < species_extent; ++ispeciesA) {
                            T_s(itor1, itor2, itor3, ispeciesA) = Ta_tmp(ispeciesA);
                        }
                    });
            };

            Kokkos::parallel_for("DoComputeMaxwellianEvolution",
                                 a_policy, kernel);
        }

        void Operator::ComputeMaxwellianEvolution(ComputeFloatType dt,
                                                  Flo64*           f5d_s) {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                                       species_extent_);

            DoComputeMaxwellianEvolution(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                         dt,
                                         a_device_visible_f5d_s, T_s_);

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
