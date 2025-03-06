// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <limits>
#include <type_traits>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/flatten.h"
#include "KOLIOP/utility/select.h"
#include "KOLIOP/utility/tordu.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        /// @brief Make Fa evolve:
        ///   f = f + dt * sum_b (nu_sab(xb) * vpar / Ta) * F_M0a * &
        ///           (Rat * Vparb - Vpara + (q_para / (na * Ta)) * (1 - 2*xa*xa/5) -
        ///           (3/5) *  Rat * (q_parb / (nb * Tb)) / (1 + (vTa/vTb)^2 ))
        ///  with Rat = sqrt(hat_Aa/hat_Ab)
        ///   xa = v /(sqrt(2*Ta)) = sqrt(2*mu*B + vpar*vpar)/(sqrt(2*Ta))
        ///   xb = xa * vTa / vTb
        ///
        template <typename ExecutionSpace,
                  typename View6D>
        static inline void
        DoComputeCpar(const Operator&  an_operator,
                      ExecutionSpace   an_execution_space,
                      ComputeFloatType dt,
                      View6D&&         f5d_s) {

            const auto& mu_extent      = an_operator.mu_extent_;
            const auto& vpar_extent    = an_operator.vpar_extent_;
            const auto& tor1_extent    = an_operator.tor1_extent_;
            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& tor3_extent    = an_operator.tor3_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& local_domain_r_offset     = an_operator.local_domain_r_offset_;
            const auto& local_domain_theta_offset = an_operator.local_domain_theta_offset_;

            const auto& hat_As        = an_operator.hat_As_;
            const auto& hat_Zs        = an_operator.hat_Zs_;
            const auto& mug           = an_operator.mug_;
            const auto& vparg         = an_operator.vparg_;
            const auto& coeff_AD_r    = an_operator.coeff_AD_r_;
            const auto& B_norm        = an_operator.B_norm_;
            const auto& n_s           = an_operator.n_s_;
            const auto& U_par_s       = an_operator.U_par_s_;
            const auto& T_s           = an_operator.T_s_;
            const auto& q_par_on_nT_s = an_operator.q_par_on_nT_s_;

            const ComputeFloatType half_dt = dt / 2.0;

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

                const ComputeFloatType hat_Aa     = hat_As(ispeciesA);
                const ComputeFloatType hat_Za     = hat_Zs(ispeciesA);
                const ComputeFloatType inv_hat_Aa = 1.0 / hat_Aa;

                const ComputeFloatType Bij = B_norm(local_domain_r_offset + ir,
                                                    local_domain_theta_offset + itheta);

                const ComputeFloatType na          = n_s(itor1, itor2, itor3, ispeciesA);
                const ComputeFloatType Vpara       = U_par_s(itor1, itor2, itor3, ispeciesA);
                const ComputeFloatType Ta          = T_s(itor1, itor2, itor3, ispeciesA);
                const ComputeFloatType q_par_on_nT = q_par_on_nT_s(itor1, itor2, itor3, ispeciesA);

                static constexpr auto k2Pi = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi;

                const ComputeFloatType vTa    = Kokkos::sqrt(Ta * inv_hat_Aa);
                const ComputeFloatType coeff1 = 1.0 / (2.0 * Ta);
                const ComputeFloatType coeff3 = na / (k2Pi * Ta * Kokkos::sqrt(k2Pi * Ta));
                const ComputeFloatType AD     = coeff_AD_r(ir);

                const ComputeFloatType inv_Ta = 1.0 / Ta;

                koliop::utility::Flatten2 flattened_mu_vpar{mu_extent, vpar_extent};

                for(OrdinalType ispeciesB = 0; ispeciesB < species_extent; ++ispeciesB) {
                    // NOTE: We do very large jump in memory, but all the
                    // threads of a team gets to load this value so in practice
                    // we observe only one access.

                    const ComputeFloatType hat_Ab         = hat_As(ispeciesB);
                    const ComputeFloatType inv_hat_Ab     = 1.0 / hat_Ab;
                    const ComputeFloatType hat_Zb         = hat_Zs(ispeciesB);
                    const ComputeFloatType nb             = n_s(itor1, itor2, itor3, ispeciesB);
                    const ComputeFloatType Vparb          = U_par_s(itor1, itor2, itor3, ispeciesB);
                    const ComputeFloatType Tb             = T_s(itor1, itor2, itor3, ispeciesB);
                    const ComputeFloatType q_parb_on_nbTb = q_par_on_nT_s(itor1, itor2, itor3, ispeciesB);

                    const ComputeFloatType Rat    = Kokkos::sqrt(hat_Aa / hat_Ab);
                    const ComputeFloatType vTb    = Kokkos::sqrt(Tb / hat_Ab);
                    const ComputeFloatType coeff2 = vTa / vTb;
                    const ComputeFloatType coeff4 = (3.0 / 5.0) * Rat * q_parb_on_nbTb / (1.0 + (Ta * hat_Ab) / (Tb * hat_Aa));

                    const ComputeFloatType factor0 = (AD * hat_Za * hat_Za * hat_Zb * hat_Zb * nb) * inv_hat_Aa * (inv_hat_Aa + inv_hat_Ab) / (vTb * vTb * vTb);
                    const ComputeFloatType factor1 = factor0 * half_dt;
                    const ComputeFloatType factor2 = Rat * Vparb - Vpara + q_par_on_nT - coeff4 + q_par_on_nT * -2.0 * 0.2;
                    const ComputeFloatType factor3 = factor1 * factor2;
                    const ComputeFloatType factor4 = factor3 * inv_Ta * coeff3;

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(a_team_handle, flattened_mu_vpar.forward()), [&](auto flattened_index) {
                        const auto mu_vpar_work = [&](auto imu, auto ivpar) {
                            // const ComputeFloatType coeff5 = vparg(ivpar) / Ta;
                            // const ComputeFloatType xa     = Kokkos::sqrt(coeff1 * (2.0 * mug(imu) * Bij + vparg(ivpar) * vparg(ivpar)));
                            // const ComputeFloatType xb     = xa * coeff2;
                            // const ComputeFloatType FM0a   = coeff3 * Kokkos::exp(-xa * xa);

                            // // Compute the function:
                            // //    Theta(x) = 3*sqrt(pi)/2 *G(x)/x
                            // //      with G(x) = (phi(x) - x phi'(x)) / (2 x^2)
                            // //      and  x = vTa_on_vTb * sqrt(E) / sqrt_Ta
                            // //      note: phi(x) = 2/sqrt(pi) int_0^x exp(-y^2) dy is computed using ERF()
                            // const ComputeFloatType Theta_func_tmp = xb > 1E-15 ?
                            //                                             koliop::portability::GyselaConstant<ComputeFloatType>::kConst_theta * ((Kokkos::erf(xb) - xb * (koliop::portability::GyselaConstant<ComputeFloatType>::kCst_phiprime * Kokkos::exp(-xb * xb))) / (2.0 * xb * xb)) / xb :
                            //                                             1.0;

                            // // Compute collision frequency nu_sab
                            // //  nu_sab(xb) = (AD * hat_Za^2 * hat_Zb^2 * Nb) * (1/hat_Aa) * &
                            // //               (1/hat_Aa + 1/hat_Ab)*Theta(xb) / vTb^3
                            // const ComputeFloatType nu_sab = (AD * hat_Za * hat_Za * hat_Zb * hat_Zb * nb) * inv_hat_Aa * (inv_hat_Aa + inv_hat_Ab) * Theta_func_tmp / (vTb * vTb * vTb);
                            // f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) += half_dt * FM0a * nu_sab * coeff5 * (Rat * Vparb - Vpara + q_par_on_nT * (1.0 - 2.0 * xa * xa * 0.2) - coeff4);

                            const ComputeFloatType xa = Kokkos::sqrt(coeff1 * (2.0 * mug(imu) * Bij + vparg(ivpar) * vparg(ivpar)));
                            const ComputeFloatType xb = xa * coeff2;

                            const ComputeFloatType Theta_func_tmp = xb > 1E-15 ?
                                                                        koliop::portability::GyselaConstant<ComputeFloatType>::kConst_theta * ((Kokkos::erf(xb) - xb * (koliop::portability::GyselaConstant<ComputeFloatType>::kCst_phiprime * Kokkos::exp(-xb * xb))) / (2.0 * xb * xb)) / xb :
                                                                        1.0;

                            f5d_s(imu, ivpar, itor1, itor2, itor3, ispeciesA) += factor4 * Kokkos::exp(-xa * xa) * Theta_func_tmp * vparg(ivpar) * xa * xa;
                        };
                        flattened_mu_vpar.Call(flattened_index, mu_vpar_work);
                    });
                }
            };

            Kokkos::parallel_for("DoComputeCpar",
                                 a_policy, kernel);
        }

        void Operator::ComputeCpar(ComputeFloatType dt,
                                   Flo64*           f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL6<Flo64>,
                                                                               koliop::utility::View::InternalInputL6>(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                                                                                       f5d_s, "f5d_s",
                                                                                                                       mu_extent_, vpar_extent_,
                                                                                                                       tor1_extent_, tor2_extent_, tor3_extent_,
                                                                                                                       species_extent_);

            {
                DoComputeCpar(*this, execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                              dt, a_device_visible_f5d_s);
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
