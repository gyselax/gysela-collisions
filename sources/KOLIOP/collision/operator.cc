// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/operator.h"

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View3D,
                  typename View1D>
        static inline void
        DoPrepareNuStar(const Operator& an_operator,
                        View3D&&        nustar_coeff,
                        const View1D&   hat_As,
                        const View1D&   hat_Zs,
                        const View1D&   rg,
                        const View1D&   nustar0_r,
                        const View1D&   q_r,
                        const View1D&   mask_buffer_r) {

            const auto a_host_visible_hat_As        = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_As);
            const auto a_host_visible_hat_Zs        = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_Zs);
            const auto a_host_visible_rg            = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), rg);
            const auto a_host_visible_nustar0_r     = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), nustar0_r);
            const auto a_host_visible_q_r           = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), q_r);
            const auto a_host_visible_mask_buffer_r = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), mask_buffer_r);
            const auto a_host_visible_nustar_coeff  = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), nustar_coeff);

            const auto& the_r_extent       = an_operator.the_r_extent_;
            const auto& the_species_extent = an_operator.the_species_extent_;

            const auto& the_local_domain_r_offset = an_operator.the_local_domain_r_offset_;

            const auto& SOL = an_operator.SOL_;
            const auto& LIM = an_operator.LIM_;

            koliop::utility::View::MHL1<ComputeFloatType> a_host_visible_mask_space{Kokkos::view_alloc(Kokkos::WithoutInitializing, "a_host_visible_mask_space"),
                                                                                    static_cast<std::size_t>(the_r_extent)};

            if(a_host_visible_q_r(0) < koliop::portability::GyselaConstant<ComputeFloatType>::kInfinity) {
                for(OrdinalType ir = 0; ir < the_r_extent; ++ir) {
                    a_host_visible_mask_space(ir) = static_cast<ComputeFloatType>(1.0) / a_host_visible_q_r(the_local_domain_r_offset + ir);
                }
            }

            if(!(LIM || SOL)) {
                for(OrdinalType ir = 0; ir < the_r_extent; ++ir) {
                    a_host_visible_mask_space(ir) *= (static_cast<ComputeFloatType>(1.0) - a_host_visible_mask_buffer_r(the_local_domain_r_offset + ir));
                }
            }

            const ComputeFloatType R0_inverse = static_cast<ComputeFloatType>(1.0) / an_operator.R0_;

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
            for(OrdinalType ispeciesB = 0; ispeciesB < the_species_extent; ++ispeciesB) {
                for(OrdinalType ispeciesA = 0; ispeciesA < the_species_extent; ++ispeciesA) {
                    const ComputeFloatType hat_Aa = a_host_visible_hat_As(ispeciesA);
                    const ComputeFloatType hat_Za = a_host_visible_hat_Zs(ispeciesA);
                    const ComputeFloatType hat_Ab = a_host_visible_hat_As(ispeciesB);
                    const ComputeFloatType hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                    const ComputeFloatType AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa + static_cast<ComputeFloatType>(1.0) / hat_Ab);

                    for(OrdinalType ir = 0; ir < the_r_extent; ++ir) {
                        // Computation of
                        // AD(r) = sqrt(2) * eps ^ (3 / 2) / (q R0_)*nustaR0_(r)
                        // which is independent on the species considered (eps = r/R0_)
                        const ComputeFloatType eps = a_host_visible_rg(the_local_domain_r_offset + ir) * R0_inverse;

                        const ComputeFloatType AD = a_host_visible_nustar0_r(the_local_domain_r_offset + ir) *
                                                    a_host_visible_mask_space(ir) *
                                                    // TODO(Etienne M): SIMPLIFY
                                                    // sqrt(2)*(rg*R0_inverse)*sqrt(rg*R0_inverse)*R0_inverse_ =
                                                    // sqrt(2)*pow(r, 5/2)*pow(rg, 3/2)
                                                    // sqrt(2)*pow(r, 5/2) is constant here.
                                                    koliop::portability::MathematicalConstant<ComputeFloatType>::kSqrt2 *
                                                    eps *
                                                    Kokkos::sqrt(eps) *
                                                    R0_inverse;

                        // Computation of nu_ab =
                        // (AD * hat_Za ^ 2 * hat_Zb ^ 2 * Nb) /
                        // (vTa ^ 2 + vTb ^ 2) ^ (3 / 2) * (1 / hat_Aa) *
                        // (1 / hat_Aa + 1 / hat_Ab)
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

        Operator::Operator(uint64_t     pSpecies_id,
                           uint64_t     the_mu_extent,
                           uint64_t     the_vpar_extent,
                           uint64_t     the_r_extent,
                           uint64_t     the_theta_extent,
                           uint64_t     the_phi_extent,
                           uint64_t     the_species_extent,
                           uint64_t     the_global_domain_r_extent,
                           uint64_t     the_global_domain_theta_extent,
                           uint64_t     the_local_domain_r_offset,
                           uint64_t     the_local_domain_theta_offset,
                           Flo64        R0,
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
                           const Flo64* rg,
                           const Flo64* nustar0_r,
                           const Flo64* q_r,
                           const Flo64* mask_buffer_r,
                           const Flo64* mask_LIM,
                           const Flo64* B_norm,
                           const Flo64* Bstar_s)
            // Constant, externally computed values
            : pSpecies_id_{static_cast<OrdinalType>(pSpecies_id)}
            , the_mu_extent_{static_cast<OrdinalType>(the_mu_extent)}
            , the_vpar_extent_{static_cast<OrdinalType>(the_vpar_extent)}
            , the_r_extent_{static_cast<OrdinalType>(the_r_extent)}
            , the_theta_extent_{static_cast<OrdinalType>(the_theta_extent)}
            , the_phi_extent_{static_cast<OrdinalType>(the_phi_extent)}
            , the_species_extent_{static_cast<OrdinalType>(the_species_extent)}
            , the_global_domain_r_extent_{static_cast<OrdinalType>(the_global_domain_r_extent)}
            , the_global_domain_theta_extent_{static_cast<OrdinalType>(the_global_domain_theta_extent)}
            , the_local_domain_r_offset_{static_cast<OrdinalType>(the_local_domain_r_offset)}
            , the_local_domain_theta_offset_{static_cast<OrdinalType>(the_local_domain_theta_offset)}
            , R0_{static_cast<ComputeFloatType>(R0)}
            , collision_interspecies_{collision_interspecies}
            , SOL_{SOL}
            , LIM_{LIM}
            , ir_SOL_separatrix_{static_cast<OrdinalType>(ir_SOL_separatrix)}
            , hat_As_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, hat_As, "hat_As_", the_species_extent_)} // NOTE: in practice, this hat_Xs contains 3 values. The size being hardcoded in global::Nspecies_max = 2
            , hat_Zs_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, hat_Zs, "hat_Zs_", the_species_extent_)} // NOTE: in practice, this hat_Xs contains 3 values. The size being hardcoded in global::Nspecies_max = 2
            , mug_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, mug, "mug_", the_mu_extent_)}
            , vparg_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, vparg, "vparg_", the_vpar_extent_)}
            , coeff_intdmu_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, coeff_intdmu, "coeff_intdmu_", the_mu_extent_)}
            , coeff_intdvpar_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, coeff_intdvpar, "coeff_intdvpar_", the_vpar_extent_)}
            , rg_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, rg, "rg_", the_global_domain_r_extent_)}
            , nustar0_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, nustar0_r, "nustar0_r_", the_global_domain_r_extent_)}
            , q_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, q_r, "q_r_", the_global_domain_r_extent_)}
            , mask_buffer_r_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL1<const Flo64>, koliop::utility::View::InternalInputL1>(Constant::DeviceExecutionSpaceType{}, mask_buffer_r, "mask_buffer_r_", the_global_domain_r_extent_)}
            , mask_LIM_{LIM_ ? koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL2<const Flo64>, koliop::utility::View::InternalInputL2>(Constant::DeviceExecutionSpaceType{}, mask_LIM, "mask_LIM_", the_global_domain_r_extent_, the_global_domain_theta_extent_) :
                               koliop::utility::View::InternalInputL2<Flo64>{}}
            , B_norm_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL2<const Flo64>, koliop::utility::View::InternalInputL2>(Constant::DeviceExecutionSpaceType{}, B_norm, "B_norm_", the_global_domain_r_extent_, the_global_domain_theta_extent_)}
            , Bstar_s_{koliop::utility::Copy::RawDataToView<koliop::utility::View::ExternalInputL4<const Flo64>, koliop::utility::View::InternalInputL4>(Constant::DeviceExecutionSpaceType{}, Bstar_s, "Bstar_s_", the_vpar_extent_, the_r_extent_, the_theta_extent_, the_species_extent_)}

            // Constant, internally computed values

            , nustar_coeff_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar_coeff_"),
                            static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_r_extent_)}

            // Variable, internally computed values

            , n_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_s_"),
                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , U_par_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_0_"),
                       static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , U_par_s_input_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_1_"),
                             static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , T_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_s_"),
                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}
            , nu_ab_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu_ab_"),
                     static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_species_extent_)}
            , q_par_on_nT_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT_s_"),
                             static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_), static_cast<std::size_t>(the_species_extent_)}

            // Miscellaneous tooling
            , the_execution_space_pool_{}

            // Miscellaneous numerical tooling
            , the_cvcd_drivers_scheduler_{} {

            // Ensure the copies above are completed.
            // Constant::DeviceExecutionSpaceType{}.fence() would also work.
            Kokkos::fence();

            DoPrepareNuStar(*this,
                            nustar_coeff_,
                            hat_As_,
                            hat_Zs_,
                            rg_,
                            nustar0_r_,
                            q_r_,
                            mask_buffer_r_);

            KOLIOP_ASSERT(the_execution_space_pool_.empty());

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
            // The null stream.
            the_execution_space_pool_.emplace_back(nullptr, true);
#endif

            if(Constant::kDisjointExecutionSpaceCount > the_execution_space_pool_.size()) {
                // Then, we use the default execution space.
                the_execution_space_pool_.emplace_back();
            }

            KOLIOP_ASSERT(!the_execution_space_pool_.empty());

            // Then we fill in with new streams.

            while(the_execution_space_pool_.size() != Constant::kDisjointExecutionSpaceCount) {
                // NOTE: We do not want to *potentially* split the resource
                // pool. Thus the repeated call to partition_space which creates
                // execution space with all the same amount of hardware
                // resources.
                // TODO(Etienne M): See what hipStreamCreateWithPriority can get
                // us.
                the_execution_space_pool_.emplace_back(Kokkos::Experimental::partition_space(the_execution_space_pool_[0], 1)[0]);
            }

            the_cvcd_drivers_scheduler_.Plan(*this,
                                             Constant::kPresetBlockExtentCount,
                                             Constant::kPresetBlockExtent);
        }

        Operator::~Operator() {
            Kokkos::fence();
        }

        void Operator::Fence(OrdinalType the_execution_space_index) const {
            the_execution_space_pool_[the_execution_space_index].fence();
        }

        void Operator::Collision(ComputeFloatType dt,
                                 Flo64*           f5d_s) {
            OrdinalType the_species[]{pSpecies_id_};
            CollisionFor(dt, f5d_s,
                         std::cbegin(the_species),
                         std::cend(the_species));
        }

        void Operator::CollisionForAllSpecies(ComputeFloatType dt,
                                              Flo64*           f5d_s) {
            std::vector<OrdinalType> the_species;
            the_species.resize(the_species_extent_);
            std::iota(std::begin(the_species), std::end(the_species), OrdinalType{});
            CollisionFor(dt, f5d_s,
                         std::cbegin(the_species),
                         std::cend(the_species));
        }

        // NOTE: It is a bit sneaky to NOT put a template in the header when the
        // template is part of the interface.
        template <typename ForwardIterator>
        void Operator::CollisionFor(ComputeFloatType dt,
                                    Flo64*           f5d_s,
                                    ForwardIterator  the_first_species,
                                    ForwardIterator  the_last_species) {

            // NOTE: When applying the operator for all collision, some
            // computation optimization can be done. We try to avoid recomputing
            // and minimizing the overhead of the operator.
            // NOTE: Each time we have a std::for_each, we could use a separate
            // stream for all function call.

            // TODO(Etienne M): We shouldn't need that for the Gysela++, yet.
            const OrdinalType collstart = 0;

            if(collstart != -1 /* TODO(Etienne M): Fortran index */) {
                ComputeMoments(f5d_s);
                PrecomputeNu();

                if((the_species_extent_ > 1) && collision_interspecies_) {
                    ComputeMaxwellianEvolution(dt, f5d_s);
                    PrecomputeNu();
                }

                ComputeQparOnNT(f5d_s);

                if(the_species_extent_ > 1) {
                    // Cpar operator
                    std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                        ComputeCpar(dt, a_species_index, f5d_s);
                    });
                }

                // Computes Fa = Fa / FM0a
                std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                    ComputeMaxwellianUnshiftedLoc(4, a_species_index, f5d_s);
                });

                // CvCd
                std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                    Apply(dt, a_species_index, f5d_s);
                });

                // Computes Fa = Fa * FM0a
                std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                    ComputeMaxwellianUnshiftedLoc(3, a_species_index, f5d_s);
                });

                if((the_species_extent_ > 1) && collision_interspecies_) {
                    // Cpar operator
                    std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                        ComputeCpar(dt, a_species_index, f5d_s);
                    });
                }

                // Correction to enforce conservation of density and fluid
                // moments
                std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                    ComputeDensityCorrection(a_species_index, f5d_s);
                });

                // ComputeUparTCorrection reads U_par_s_input_ and outputs to
                // U_par_s_. This is required if more than 1 species is used, we
                // can not write in U_par_s_ while it is still being read (by
                // other threads, species, thread block, etc.).
                std::swap(U_par_s_, U_par_s_input_);
                std::for_each(the_first_species, the_last_species, [&](OrdinalType a_species_index) {
                    ComputeUparTCorrection(dt, a_species_index, U_par_s_input_, f5d_s);
                });

                // TODO(Etienne M): We may have an issue:
                // - ComputeMaxwellianEvolution reads U_par_s for all species;
                // - We may write a new U_par_s for  all species in ComputeUparTCorrection;
                // - In the past, only the pSpecies was modified in ComputeUparTCorrection;
                //      - Did we miss a reduction over species ?

                if((the_species_extent_ > 1) && collision_interspecies_) {
                    ComputeMaxwellianEvolution(dt, f5d_s);
                    PrecomputeNu();
                }
            }
        }

    } // namespace collision
} // namespace koliop
