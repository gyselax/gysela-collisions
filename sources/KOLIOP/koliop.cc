// SPDX-License-Identifier: MIT

#include "KOLIOP/koliop.h"

#include <Kokkos_Core.hpp>

#include "KOLIOP/collision/policy.h"
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

::koliop_Status koliop_Create(::koliop_State* a_state_handle) {
    return koliop::error::FromException([&]() {
        if(a_state_handle == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State<koliop::collision::CollisionPolicy>;

        static constexpr int  Npolmax                = 3;
        static constexpr int  index_max              = 4;
        static constexpr int  Nmu                    = 47;
        static constexpr int  Nvpar                  = 127;
        static constexpr int  Nspecies               = 1;
        static constexpr int  pSpecies_id            = 0;
        static constexpr int  mu_id                  = 4; // [0, nproc)
        static constexpr int  istart                 = 0;
        static constexpr int  iend                   = 31;
        static constexpr int  jstart                 = 0;
        static constexpr int  jend                   = 31;
        static constexpr int  collstart              = 0; // [0, nproc)
        static constexpr int  collend                = 0; // [0, nproc)
        static constexpr int  R0                     = 320.;
        static constexpr bool collision_interspecies = true;
        static constexpr bool SOL                    = false;
        static constexpr bool LIM                    = false;

        // lbound(comb_mat,1) = 0;
        // ubound(comb_mat,1) = 5;
        // size(comb_mat,1) = 6;

        // lbound(comb_mat,2) = 0;
        // ubound(comb_mat,2) = 5;
        // size(comb_mat,2) = 6;

        // lbound(hat_As,1) = 0;
        // ubound(hat_As,1) = 2;
        // size(hat_As,1) = 3;

        // lbound(hat_Zs,1) = 0;
        // ubound(hat_Zs,1) = 2;
        // size(hat_Zs,1) = 3;

        // lbound(mug,1) = 0;
        // ubound(mug,1) = 47;
        // size(mug,1) = 48;

        // lbound(vparg,1) = 0;
        // ubound(vparg,1) = 127;
        // size(vparg,1) = 128;

        // lbound(coeff_intdmu,1) = 0;
        // ubound(coeff_intdmu,1) = 47;
        // size(coeff_intdmu,1) = 48;

        // lbound(coeff_intdvpar,1) = 0;
        // ubound(coeff_intdvpar,1) = 127;
        // size(coeff_intdvpar,1) = 128;

        koliop::Flo64        dummy[1024];
        const koliop::Flo64* comb_mat       = dummy;
        const koliop::Flo64* hat_As         = dummy;
        const koliop::Flo64* hat_Zs         = dummy;
        const koliop::Flo64* mug            = dummy;
        const koliop::Flo64* vparg          = dummy;
        const koliop::Flo64* coeff_intdmu   = dummy;
        const koliop::Flo64* coeff_intdvpar = dummy;

        auto* const a_state = new State{Npolmax,
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
                                        collision_interspecies,
                                        SOL,
                                        LIM,
                                        comb_mat, 6, 6,
                                        hat_As, 3,
                                        hat_Zs, 3,
                                        mug, 48,
                                        vparg, 128,
                                        coeff_intdmu, 48,
                                        coeff_intdvpar, 128};

        KOLIOP_ASSERT(a_state != nullptr);

        *a_state_handle = reinterpret_cast<koliop_State>(a_state);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Release(::koliop_State a_state) {
    return koliop::error::FromException([&]() {
        using State = koliop::collision::State<koliop::collision::CollisionPolicy>;

        // Freeing a nullptr is *fine*.
        delete reinterpret_cast<State*>(a_state);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Compute(::koliop_State a_state) {
    return koliop::error::FromException([&]() {
        if(a_state == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        // using State = koliop::collision::State<koliop::collision::CollisionPolicy>;
        // reinterpret_cast<State*>(a_state)->Multiply();

        return KOLIOP_STATUS_SUCCESS;
    });
}
