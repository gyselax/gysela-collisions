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

        using State         = koliop::collision::State<koliop::collision::CollisionPolicy>;
        auto* const a_state = new State{};

        KOLIOP_ASSERT(a_state != nullptr);

        *a_state_handle = reinterpret_cast<koliop_State>(a_state);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Release(::koliop_State a_state) {
    return koliop::error::FromException([&]() {
        if(a_state == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using State = koliop::collision::State<koliop::collision::CollisionPolicy>;
        delete reinterpret_cast<State*>(a_state);

        return KOLIOP_STATUS_SUCCESS;
    });
}
