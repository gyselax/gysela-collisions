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
                              uint64_t           comb_mat_extent_0,
                              uint64_t           comb_mat_extent_1,
                              const double*      hat_As,
                              uint64_t           hat_As_extent_0,
                              const double*      hat_Zs,
                              uint64_t           hat_Zs_extent_0) {
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
                                            comb_mat, comb_mat_extent_0, comb_mat_extent_1,
                                            hat_As, hat_As_extent_0,
                                            hat_Zs, hat_Zs_extent_0};

        KOLIOP_ASSERT(an_operator != nullptr);

        *an_operator_handle = reinterpret_cast<koliop_Operator>(an_operator);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Release(::koliop_Operator an_operator) {
    return koliop::error::FromException([&]() {
        using State = koliop::collision::State;

        // Freeing a nullptr is *fine*.
        delete reinterpret_cast<State*>(an_operator);

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
        reinterpret_cast<State*>(an_operator)->PrepareNuStar(rg, nustaR0_r, q_r, mask_buffer_r, nustar_coeff);

        return KOLIOP_STATUS_SUCCESS;
    });
}
