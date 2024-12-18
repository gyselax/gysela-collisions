// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/tordu.h"

namespace koliop {
    namespace collision {

        template <typename View3D,
                  typename View1D>
        static inline void
        DoPrepareNuStar(const Operator& an_operator,
                        View3D&&        nustar_coeff,
                        const View1D&   hat_As,
                        const View1D&   hat_Zs,
                        const View1D&   coeff_AD_r) {

            const auto a_host_visible_nustar_coeff = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), nustar_coeff);

            const auto a_host_visible_hat_As     = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_As);
            const auto a_host_visible_hat_Zs     = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_Zs);
            const auto a_host_visible_coeff_AD_r = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), coeff_AD_r);

            const auto& r_extent       = an_operator.r_extent_;
            const auto& species_extent = an_operator.species_extent_;

            const auto& local_domain_r_offset = an_operator.local_domain_r_offset_;

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
            for(OrdinalType ispeciesB = 0; ispeciesB < species_extent; ++ispeciesB) {
                for(OrdinalType ispeciesA = 0; ispeciesA < species_extent; ++ispeciesA) {
                    const ComputeFloatType hat_Aa = a_host_visible_hat_As(ispeciesA);
                    const ComputeFloatType hat_Za = a_host_visible_hat_Zs(ispeciesA);
                    const ComputeFloatType hat_Ab = a_host_visible_hat_As(ispeciesB);
                    const ComputeFloatType hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                    const ComputeFloatType AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa + static_cast<ComputeFloatType>(1.0) / hat_Ab);

                    for(OrdinalType ir = 0; ir < r_extent; ++ir) {
                        // AD(r) = sqrt(2)*eps(r)^(3/2)/(q(r) R0_)*nustaR0_(r) which
                        // is independent on the species considered
                        // (eps(r) = r/R0_).
                        // AD comprises a mask in r direction to avoid to *
                        // compute collision in the scrape-off layer.
                        const ComputeFloatType AD = a_host_visible_coeff_AD_r(local_domain_r_offset + ir);

                        // Computation of nu_ab =
                        //  (AD * hat_Za^2 * hat_Zb^2 * Nb) /
                        //  (vTa^2 + vTb^2)^(3/2) *
                        //  (1/hat_Aa) * (1/hat_Aa + 1/hat_Ab)
                        a_host_visible_nustar_coeff(ispeciesA, ispeciesB, ir) = AD * AD_factor;
                    }
                }
            }

            koliop::utility::Copy::Shallow(Constant::DeviceExecutionSpaceType{},
                                           a_host_visible_nustar_coeff,
                                           nustar_coeff);
        }


        ////////////////////////////////////////////////////////////////////////
        // Operator methods definition
        ////////////////////////////////////////////////////////////////////////

        Operator::Operator(uint64_t     mu_extent,
                           uint64_t     vpar_extent,
                           uint64_t     tor1_extent,
                           uint64_t     tor2_extent,
                           uint64_t     tor3_extent,
                           uint64_t     species_extent,
                           uint64_t     global_domain_r_extent,
                           uint64_t     global_domain_theta_extent,
                           uint64_t     local_domain_r_offset,
                           uint64_t     local_domain_theta_offset,
                           bool         collision_interspecies,
                           bool         SOL,
                           bool         LIM,
                           uint64_t     ir_SOL_separatrix,
                           const Flo64* hat_As,
                           const Flo64* hat_Zs,
                           const Flo64* mug,
                           const Flo64* vparg,
                           const Flo64* coeff_intdmu,
                           const Flo64* coeff_intdvpar,
                           const Flo64* coeff_AD_r,
                           const Flo64* mask_buffer_r,
                           const Flo64* mask_LIM,
                           const Flo64* B_norm,
                           const Flo64* Bstar_s)
            // Constant, externally computed values
            : mu_extent_{static_cast<OrdinalType>(mu_extent)}
            , vpar_extent_{static_cast<OrdinalType>(vpar_extent)}
            , tor1_extent_{static_cast<OrdinalType>(tor1_extent)}
            , tor2_extent_{static_cast<OrdinalType>(tor2_extent)}
            , tor3_extent_{static_cast<OrdinalType>(tor3_extent)}
            , r_extent_{koliop::utility::GetRValue(tor1_extent_, tor2_extent_, tor3_extent_)}
            , theta_extent_{koliop::utility::GetThetaValue(tor1_extent_, tor2_extent_, tor3_extent_)}
            , phi_extent_{koliop::utility::GetPhiValue(tor1_extent_, tor2_extent_, tor3_extent_)}
            , species_extent_{static_cast<OrdinalType>(species_extent)}
            , global_domain_r_extent_{static_cast<OrdinalType>(global_domain_r_extent)}
            , global_domain_theta_extent_{static_cast<OrdinalType>(global_domain_theta_extent)}
            , local_domain_r_offset_{static_cast<OrdinalType>(local_domain_r_offset)}
            , local_domain_theta_offset_{static_cast<OrdinalType>(local_domain_theta_offset)}
            , collision_interspecies_{collision_interspecies}
            , SOL_{SOL}
            , LIM_{LIM}
            , ir_SOL_separatrix_{static_cast<OrdinalType>(ir_SOL_separatrix)}
            , hat_As_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, hat_As, "hat_As_", species_extent_)} // NOTE: in practice, this hat_Xs contains 3 values. The size being hardcoded in global::Nspecies_max = 2
            , hat_Zs_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, hat_Zs, "hat_Zs_", species_extent_)} // NOTE: in practice, this hat_Xs contains 3 values. The size being hardcoded in global::Nspecies_max = 2
            , mug_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, mug, "mug_", mu_extent_)}
            , vparg_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, vparg, "vparg_", vpar_extent_)}
            , coeff_intdmu_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, coeff_intdmu, "coeff_intdmu_", mu_extent_)}
            , coeff_intdvpar_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, coeff_intdvpar, "coeff_intdvpar_", vpar_extent_)}
            , coeff_AD_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, coeff_AD_r, "coeff_AD_r_", global_domain_r_extent_)}
            , mask_buffer_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, mask_buffer_r, "mask_buffer_r_", global_domain_r_extent_)}
            , mask_LIM_{LIM_ ? koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL2<const Flo64>, koliop::utility::View::InternalInputL2>(Constant::DeviceExecutionSpaceType{}, mask_LIM, "mask_LIM_", global_domain_r_extent_, global_domain_theta_extent_) :
                               koliop::utility::View::InternalInputL2<Flo64>{}}
            , B_norm_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL2<const Flo64>, koliop::utility::View::InternalInputL2>(Constant::DeviceExecutionSpaceType{}, B_norm, "B_norm_", global_domain_r_extent_, global_domain_theta_extent_)}
            , Bstar_s_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL4<const Flo64>, koliop::utility::View::InternalInputL4>(Constant::DeviceExecutionSpaceType{}, Bstar_s, "Bstar_s_", vpar_extent_, r_extent_, theta_extent_, species_extent_)}

            // Constant, internally computed values

            , nustar_coeff_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar_coeff_"),
                            static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(r_extent_)}

            // Variable, internally computed values

            , n_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_s_"),
                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , U_par_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_0_"),
                       static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , U_par_s_input_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_1_"),
                             static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , T_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_s_"),
                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , nu_ab_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu_ab_"),
                     static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_)}
            , q_par_on_nT_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT_s_"),
                             static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}

            // Miscellaneous tooling
            , execution_space_pool_{}

            // Miscellaneous numerical tooling
            , cvcd_drivers_scheduler_{} {

            // Ensure the copies above are completed.
            // Constant::DeviceExecutionSpaceType{}.fence() would also work.
            Kokkos::fence();

            DoPrepareNuStar(*this,
                            nustar_coeff_,
                            hat_As_,
                            hat_Zs_,
                            coeff_AD_r_);

            KOLIOP_ASSERT(execution_space_pool_.empty());

            // We emplace the the default execution space.
            Constant::DeviceExecutionSpaceType default_execution_space{};

            while(execution_space_pool_.size() != Constant::kDisjointExecutionSpaceCount) {
                // TODO(Etienne M): See what hipStreamCreateWithPriority can get
                // reduce the performance.
                execution_space_pool_.emplace_back(Kokkos::Experimental::partition_space(default_execution_space, 1)[0]);
            }

            cvcd_drivers_scheduler_.Plan(*this,
                                         Constant::kPresetBlockExtentCount,
                                         Constant::kPresetBlockExtent);
        }

        Operator::~Operator() {
            Kokkos::fence();
        }

        void Operator::Fence(OrdinalType execution_space_index) const {
            KOLIOP_ASSERT(execution_space_index < Constant::kDisjointExecutionSpaceCount);
            execution_space_pool_[execution_space_index].fence();
        }

        void Operator::FenceAll() const {
            for(auto&& an_execution_space : execution_space_pool_) {
                an_execution_space.fence();
            }

            // TODO(Etienne M): Do we full fence or wait for all the streams ?
            // Fence is a lot cheaper it seems. Though if Gysela executes other
            // GPU kernel while koliop works, this could slow things down.
        }

        void Operator::Collision(ComputeFloatType dt,
                                 Flo64*           f5d_s) {
            // Assume all our streams are empty.

            ComputeMoments(f5d_s);
            PrecomputeNu();

            if((species_extent_ > 1) && collision_interspecies_) {
                ComputeMaxwellianEvolution(dt, f5d_s);
                PrecomputeNu();
            }

            ComputeQparOnNT(f5d_s);

            if(species_extent_ > 1) {
                // Cpar operator
                ComputeCpar(dt, f5d_s);
            }

            // Computes Fa = Fa / FM0a
            ComputeMaxwellianUnshiftedLoc(4, f5d_s);

            // Implied Fence(Constant::kDefaultExecutionSpacePoolIndex);

            // CvCd
            Apply(dt, f5d_s);

            // Implied FenceAll();

            // Computes Fa = Fa * FM0a
            ComputeMaxwellianUnshiftedLoc(3, f5d_s);

            if((species_extent_ > 1) && collision_interspecies_) {
                // Cpar operator
                ComputeCpar(dt, f5d_s);
            }

            // Correction to enforce conservation of density and fluid
            // moments
            ComputeDensityCorrection(f5d_s);

            ComputeUparTCorrection(dt, f5d_s);

            if((species_extent_ > 1) && collision_interspecies_) {
                ComputeMaxwellianEvolution(dt, f5d_s);
                PrecomputeNu();
            }

            // Necessary user Fence(Constant::kDefaultExecutionSpacePoolIndex);
        }

    } // namespace collision
} // namespace koliop
