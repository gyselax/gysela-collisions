// SPDX-License-Identifier: MIT

#include "KOLIOP/koliop.h"

#include <Kokkos_Core.hpp>

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/error/exception.h"


////////////////////////////////////////////////////////////////////////////////
// koliop method definition
////////////////////////////////////////////////////////////////////////////////

::koliop_Status koliop_Initialize(uint32_t a_host_thread_count,
                                  uint32_t a_device_index,
                                  int32_t  with_verbose_initialization) {
    return koliop::error::FromException([&]() {
        if(Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        Kokkos::initialize(Kokkos::InitializationSettings()
                               .set_num_threads(static_cast<int>(a_host_thread_count))
                               .set_device_id(static_cast<int>(a_device_index))
                               .set_disable_warnings(!with_verbose_initialization)
                               // NOTE: As of version 4.0.0
                               // set_print_configuration does not work ?
                               .set_print_configuration(with_verbose_initialization));

        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(with_verbose_initialization) {
            Kokkos::print_configuration(std::cout, false);
        }

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Deinitialize(void) {
    return koliop::error::FromException([]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_DEINITIALIZATION_FAILURE;
        }

        Kokkos::finalize();

        if(Kokkos::is_initialized()) {
            return KOLIOP_STATUS_DEINITIALIZATION_FAILURE;
        }

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Create(::koliop_Operator* an_operator_handle,
                              uint64_t           Npolmax,
                              uint64_t           index_max,
                              uint64_t           pSpecies_id,
                              uint64_t           mu_id,
                              uint64_t           the_mu_extent,
                              uint64_t           the_vpar_extent,
                              uint64_t           the_r_extent,
                              uint64_t           the_theta_extent,
                              uint64_t           the_phi_extent,
                              uint64_t           the_species_extent,
                              double             R0,
                              uint32_t           collision_interspecies,
                              uint32_t           SOL,
                              uint32_t           LIM,
                              uint64_t           ir_SOL_separatrix,
                              double             dt,
                              const double*      comb_mat,
                              const double*      hat_As,
                              const double*      hat_Zs,
                              const double*      mug,
                              const double*      vparg,
                              const double*      coeff_intdmu,
                              const double*      coeff_intdvpar,
                              const double*      rg,
                              const double*      nustar0_r,
                              const double*      q_r,
                              const double*      mask_buffer_r,
                              const double*      mask_LIM,
                              const double*      B_norm,
                              const double*      Bstar_s) {
    return koliop::error::FromException([&]() {
        using State = koliop::collision::State;

        if(an_operator_handle == nullptr ||
           // We expect Npolmax to be constant.
           Npolmax != State::kMaximumLaguerrePolynomialDegree ||
           (index_max != 2 * (Npolmax - 1)) ||
           comb_mat == nullptr ||
           hat_As == nullptr ||
           hat_Zs == nullptr ||
           mug == nullptr ||
           vparg == nullptr ||
           coeff_intdmu == nullptr ||
           coeff_intdvpar == nullptr ||
           rg == nullptr ||
           nustar0_r == nullptr ||
           q_r == nullptr ||
           mask_buffer_r == nullptr ||
           // TODO(Etienne M): Is that nullptr when LIM is false ?
           (static_cast<bool>(LIM) && mask_LIM == nullptr) ||
           B_norm == nullptr ||
           Bstar_s == nullptr) {
            // TODO(Etienne M): More checks
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        auto* const an_operator = new State{pSpecies_id,
                                            mu_id,
                                            the_mu_extent,
                                            the_vpar_extent,
                                            the_r_extent,
                                            the_theta_extent,
                                            the_phi_extent,
                                            the_species_extent,
                                            R0,
                                            static_cast<bool>(collision_interspecies),
                                            static_cast<bool>(SOL),
                                            static_cast<bool>(LIM),
                                            ir_SOL_separatrix,
                                            dt,
                                            koliop::collision::ViewUHL2<const koliop::Flo64>{// "unmanaged_host_comb_mat",
                                                                                             comb_mat, ((index_max + 1) + 1) - 0, ((index_max + 1) + 1) - 0},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_hat_As",
                                                                                             hat_As, Npolmax},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_hat_Zs",
                                                                                             hat_Zs, Npolmax},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_mug",
                                                                                             mug, the_mu_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_vparg",
                                                                                             vparg, the_vpar_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_coeff_intdmu",
                                                                                             coeff_intdmu, the_mu_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_coeff_intdvpar",
                                                                                             coeff_intdvpar, the_vpar_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_rg",
                                                                                             rg, the_r_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_nustar0_r",
                                                                                             nustar0_r, the_r_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_q_r",
                                                                                             q_r, the_r_extent},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_mask_buffer_r",
                                                                                             mask_buffer_r, the_r_extent},
                                            static_cast<bool>(LIM) ? koliop::collision::ViewUHL2<const koliop::Flo64>{// "unmanaged_host_mask_LIM",
                                                                                                                      mask_LIM, the_r_extent, the_theta_extent} :
                                                                     koliop::collision::ViewUHL2<const koliop::Flo64>{/* "unmanaged_host_mask_LIM" */},
                                            koliop::collision::ViewUHL2<const koliop::Flo64>{// "unmanaged_host_B_norm",
                                                                                             B_norm, the_r_extent, the_theta_extent + 1 /* Weird +1 */},
                                            koliop::collision::ViewUHL4<const koliop::Flo64>{// "unmanaged_host_Bstar_s",
                                                                                             Bstar_s, the_vpar_extent, the_r_extent, the_theta_extent, the_species_extent}};

        KOLIOP_ASSERT(an_operator != nullptr);

        *an_operator_handle = reinterpret_cast<koliop_Operator>(an_operator);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Release(::koliop_Operator an_operator) {
    return koliop::error::FromException([&]() {
        using State = koliop::collision::State;

        // Freeing a nullptr is *fine*.
        delete reinterpret_cast<const State*>(an_operator);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMoments(::koliop_Operator an_operator,
                                      double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<State*>(an_operator)->ComputeMoments(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_PrecomputeNu(::koliop_Operator an_operator) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<State*>(an_operator)->PrecomputeNu();

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMaxwellianEvolution(::koliop_Operator an_operator,
                                                  double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<State*>(an_operator)->ComputeMaxwellianEvolution(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeQparOnNT(::koliop_Operator an_operator,
                                       const double*     f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<State*>(an_operator)->ComputeQparOnNT(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeCpar(::koliop_Operator an_operator,
                                   double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeCpar(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMaxwellianUnshiftedLoc(::koliop_Operator an_operator,
                                                     double*           f5d_s,
                                                     uint32_t          choice) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr ||
           !(choice == 3 || choice == 4)) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeMaxwellianUnshiftedLoc(f5d_s, choice);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Apply(::koliop_Operator an_operator,
                             double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->Apply(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeDensityCorrection(::koliop_Operator an_operator,
                                                double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeDensityCorrection(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeUparTCorrection(::koliop_Operator an_operator,
                                              double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<State*>(an_operator)->ComputeUparTCorrection(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}
