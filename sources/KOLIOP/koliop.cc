// SPDX-License-Identifier: MIT

#include "KOLIOP/koliop.h"

#include <Kokkos_Core.hpp>

#include "KOLIOP/collision/state.h"
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
                               .set_print_configuration(with_verbose_initialization));

        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
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
                              uint64_t           Nmu,
                              uint64_t           Nvpar,
                              uint64_t           Nspecies,
                              uint64_t           pSpecies_id,
                              uint64_t           mu_id,
                              uint64_t           istart,
                              uint64_t           iend,
                              uint64_t           jstart,
                              uint64_t           jend,
                              int64_t            collstart,
                              int64_t            collend,
                              double             R0,
                              uint32_t           collision_interspecies,
                              uint32_t           SOL,
                              uint32_t           LIM,
                              const double*      comb_mat,
                              const double*      hat_As,
                              const double*      hat_Zs,
                              const double*      mug,
                              const double*      vparg,
                              const double*      coeff_intdmu,
                              const double*      coeff_intdvpar) {
    return koliop::error::FromException([&]() {
        if(an_operator_handle == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;

        auto* const an_operator = new State{Npolmax,
                                            index_max,
                                            Nmu,
                                            Nvpar,
                                            Nspecies,
                                            pSpecies_id,
                                            mu_id,
                                            istart,
                                            iend,
                                            jstart,
                                            jend,
                                            collstart,
                                            collend,
                                            R0,
                                            static_cast<bool>(collision_interspecies),
                                            static_cast<bool>(SOL),
                                            static_cast<bool>(LIM),
                                            koliop::collision::ViewUHL2<const koliop::Flo64>{// "unmanaged_host_comb_mat",
                                                                                             comb_mat, ((index_max + 1) + 1) - 0, ((index_max + 1) + 1) - 0},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_hat_As",
                                                                                             hat_As, Npolmax},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_hat_Zs",
                                                                                             hat_Zs, Npolmax},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_mug",
                                                                                             mug, Nmu + 1},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_vparg",
                                                                                             vparg, Nvpar + 1},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_coeff_intdmu",
                                                                                             coeff_intdmu, Nmu + 1},
                                            koliop::collision::ViewUHL1<const koliop::Flo64>{// "unmanaged_host_coeff_intdvpar",
                                                                                             coeff_intdvpar, Nvpar + 1}};

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

::koliop_Status koliop_PrepareNuStar(::koliop_Operator an_operator,
                                     const double*     rg,
                                     const double*     nustaR0_r,
                                     const double*     q_r,
                                     const double*     mask_buffer_r,
                                     double*           nustar_coeff) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->PrepareNuStar(rg, nustaR0_r, q_r, mask_buffer_r, nustar_coeff);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMoments(::koliop_Operator an_operator,
                                      const double*     B_norm,
                                      const double*     Bstar_s,
                                      double*           f5d_s,
                                      double*           n_s,
                                      double*           U_par_s,
                                      double*           T_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeMoments(B_norm, Bstar_s, f5d_s, n_s, U_par_s, T_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeFrequencyNUAB(::koliop_Operator an_operator,
                                            double            nustar_coeff,
                                            uint64_t          speciesA,
                                            uint64_t          speciesB,
                                            double            Ta,
                                            double            nb,
                                            double            Tb,
                                            double*           nu_ab) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeFrequencyNUAB(nustar_coeff, speciesA, speciesB, Ta, nb, Tb, nu_ab);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeEvolutionMaxwellian(::koliop_Operator an_operator,
                                                  const double*     nustar_coeff,
                                                  const double*     B_norm,
                                                  double*           f5d_s,
                                                  const double*     n_s,
                                                  const double*     U_par_s,
                                                  double*           T_s,
                                                  double            dt) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeEvolutionMaxwellian(nustar_coeff, B_norm, f5d_s, n_s, U_par_s, T_s, dt);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeQparOnNT(::koliop_Operator an_operator,
                                       const double*     B_norm,
                                       const double*     Bstar_s,
                                       const double*     f5d_s,
                                       const double*     n_s,
                                       const double*     U_par_s,
                                       const double*     T_s,
                                       double*           q_par_on_nT_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeQparOnNT(B_norm, Bstar_s, f5d_s, n_s, U_par_s, T_s, q_par_on_nT_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeCpar(::koliop_Operator an_operator,
                                   const double*     rg,
                                   const double*     B_norm,
                                   const double*     q_r,
                                   const double*     nustaR0_r,
                                   double*           f5d_s,
                                   const double*     n_s,
                                   const double*     U_par_s,
                                   const double*     T_s,
                                   const double*     q_par_on_nT_s,
                                   double            dt) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeCpar(rg, B_norm, q_r, nustaR0_r, f5d_s, n_s, U_par_s, T_s, q_par_on_nT_s, dt);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeOperationMaxwellianUnshiftedLoc(::koliop_Operator an_operator,
                                                              const double*     B_norm,
                                                              double*           f5d_s,
                                                              const double*     n_s,
                                                              const double*     T_s,
                                                              uint32_t          choice) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeOperationMaxwellianUnshiftedLoc(B_norm, f5d_s, n_s, T_s, choice);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeCorrectionDensity(::koliop_Operator an_operator,
                                                const double*     Bstar_s,
                                                double*           f5d_s,
                                                const double*     n_s) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeCorrectionDensity(Bstar_s, f5d_s, n_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeCorrectionUparT(::koliop_Operator an_operator,
                                              const double*     nustar_coeff,
                                              const double*     B_norm,
                                              const double*     Bstar_s,
                                              double*           f5d_s,
                                              const double*     n_s,
                                              double*           U_par_s,
                                              const double*     T_s,
                                              const double*     q_par_on_nT_s,
                                              double            dt) {
    return koliop::error::FromException([&]() {
        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State;
        reinterpret_cast<const State*>(an_operator)->ComputeCorrectionUparT(nustar_coeff, B_norm, Bstar_s, f5d_s, n_s, U_par_s, T_s, q_par_on_nT_s, dt);

        return KOLIOP_STATUS_SUCCESS;
    });
}
