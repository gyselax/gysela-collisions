// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

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

            const auto& pSpecies_id = an_operator.pSpecies_id_;

            const auto the_kernel = KOKKOS_LAMBDA(auto ir, auto itheta, auto iphi) {
                // TODO(Etienne M): INDEX not coalesced. Flatten more ?

                ComputeFloatType na = 0.0;
                for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    const ComputeFloatType jacob_vel = koliop::portability::MathematicalConstant<ComputeFloatType>::k2Pi * Bstar_s(ivpar, ir, itheta, pSpecies_id);
                    for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                        const ComputeFloatType ftmp = Kokkos::max(f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id),
                                                                  0.0);

                        na += jacob_vel * coeff_intdmu(imu) * coeff_intdvpar(ivpar) * ftmp;

                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) = ftmp;
                    }
                }

                const ComputeFloatType na_inverse = 1.0 / na;

                for(OrdinalType ivpar = 0; ivpar < the_vpar_extent; ++ivpar) {
                    for(OrdinalType imu = 0; imu < the_mu_extent; ++imu) {
                        f5d_s(imu, ivpar, ir, itheta, iphi, pSpecies_id) *= n_s(ir, itheta, iphi, pSpecies_id) * na_inverse;
                    }
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3,
                                                                     Kokkos::Iterate::Left,
                                                                     Kokkos::Iterate::Left>>{{0, 0, 0},
                                                                                             {the_r_extent,
                                                                                              the_theta_extent,
                                                                                              the_phi_extent}};

            Kokkos::parallel_for("DoComputeDensityCorrection",
                                 a_policy, the_kernel);
        }

        void Operator::ComputeDensityCorrection(Flo64* f5d_s) const {

            auto a_device_visible_f5d_s = koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>,
                                                                               koliop::utility::View::MDL6>(f5d_s, "f5d_s",
                                                                                                            the_mu_extent_, the_vpar_extent_,
                                                                                                            the_r_extent_, the_theta_extent_, the_phi_extent_,
                                                                                                            the_species_extent_);

            DoComputeDensityCorrection(*this, a_device_visible_f5d_s);

            koliop::utility::Copy::Shallow(a_device_visible_f5d_s,
                                           koliop::utility::View::UHL6<Flo64>{f5d_s,
                                                                              static_cast<std::size_t>(the_mu_extent_), static_cast<std::size_t>(the_vpar_extent_),
                                                                              static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                              static_cast<std::size_t>(the_species_extent_)});

            // TODO(Etienne M): ASYNC, relax that constraint ?
            Kokkos::fence();
        }

    } // namespace collision
} // namespace koliop
