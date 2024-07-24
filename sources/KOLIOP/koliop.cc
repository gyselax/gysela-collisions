// SPDX-License-Identifier: MIT

#include "KOLIOP/koliop.h"

#include <Kokkos_Core.hpp>
#include <iostream>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/error/exception.h"
#include "KOLIOP/internal/configuration.h"


////////////////////////////////////////////////////////////////////////////////
// koliop method definition
////////////////////////////////////////////////////////////////////////////////

::koliop_Status koliop_Initialize(int8_t do_kokkos_initialization,
                                  size_t a_host_thread_count,
                                  size_t a_device_index,
                                  int8_t with_verbose_initialization) {
    return koliop::error::FromException([&]() {
        if(static_cast<bool>(do_kokkos_initialization) == Kokkos::is_initialized()) {
            // Either we are asked to initialize Kokkos and it is already
            // initialized or we are asked not to initialize kokkos and it was
            // done already (XNOR).
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(static_cast<bool>(with_verbose_initialization)) {
            std::cout << "Koliop: v" << KOLIOP_VERSION << "\n";
            std::cout << "  KOLIOP_TOOLCHAIN_NAME: " << KOLIOP_TOOLCHAIN_NAME << "\n";
#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            std::cout << "  KOLIOP_FENCE_ON_OPERATOR_EXIT: true\n";
#else
            std::cout << "  KOLIOP_FENCE_ON_OPERATOR_EXIT: false\n";
#endif

#if defined(KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE)
            std::cout << "  KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: true\n";
#else
            std::cout << "  KOLIOP_ASSUME_INPUT_BUFFERS_ARE_DEVICE_COMPATIBLE: false\n";
#endif
            std::cout << "  koliop::collision::Constant::kDisjointExecutionSpaceCount: " << koliop::collision::Constant::kDisjointExecutionSpaceCount << "\n";
            std::cout << "  koliop::collision::Constant::kDefaultExecutionSpacePoolIndex: " << koliop::collision::Constant::kDefaultExecutionSpacePoolIndex << "\n";
            std::cout << "  koliop::collision::Constant::kPresetBlockExtent: ";
            for(std::size_t i = 0; i < koliop::collision::Constant::kPresetBlockExtentCount; ++i) {
                std::cout << "[" << koliop::collision::Constant::kPresetBlockExtent[i][0]
                          << "," << koliop::collision::Constant::kPresetBlockExtent[i][1]
                          << "," << koliop::collision::Constant::kPresetBlockExtent[i][2]
                          << "] ";
            }
            std::cout << "\n";
        }

        if(static_cast<bool>(do_kokkos_initialization)) {
            Kokkos::initialize(Kokkos::InitializationSettings()
                                   .set_num_threads(static_cast<int>(a_host_thread_count))
                                   .set_device_id(static_cast<int>(a_device_index))
                                   .set_disable_warnings(!static_cast<bool>(with_verbose_initialization))
                                   // NOTE: As of version 4.0.0
                                   // set_print_configuration does not work ?
                                   .set_print_configuration(static_cast<bool>(with_verbose_initialization)));
        }

        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(static_cast<bool>(with_verbose_initialization)) {
            Kokkos::print_configuration(std::cout, false);
        }

        std::cout << std::flush;

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Deinitialize(int8_t do_kokkos_deinitialization) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_DEINITIALIZATION_FAILURE;
        }

        if(static_cast<bool>(do_kokkos_deinitialization)) {
            Kokkos::finalize();

            if(Kokkos::is_initialized()) {
                return KOLIOP_STATUS_DEINITIALIZATION_FAILURE;
            }
        }

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Create(::koliop_Operator* an_operator_handle,
                              size_t  the_mu_extent,
                              size_t  the_vpar_extent,
                              size_t  the_r_extent,
                              size_t  the_theta_extent,
                              size_t  the_phi_extent,
                              size_t  the_species_extent,
                              size_t  the_global_domain_r_extent,
                              size_t  the_global_domain_theta_extent,
                              size_t  the_local_domain_r_offset,
                              size_t  the_local_domain_theta_offset,
                              double  R0,
                              int8_t  collision_interspecies,
                              int8_t  SOL,
                              int8_t  LIM,
                              int64_t ir_SOL_separatrix,
                              const double*,
                              const double* hat_As,
                              const double* hat_Zs,
                              const double* mug,
                              const double* vparg,
                              const double* coeff_intdmu,
                              const double* coeff_intdvpar,
                              const double* rg,
                              const double* nustar0_r,
                              const double* q_r,
                              const double* mask_buffer_r,
                              const double* mask_LIM,
                              const double* B_norm,
                              const double* Bstar_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator_handle == nullptr ||
           !(0 <= ir_SOL_separatrix && static_cast<size_t>(ir_SOL_separatrix) <= the_global_domain_r_extent) ||
           // We store a constant expression of the binomial coeffs. We do not need someone to given them to us.
           // comb_mat == nullptr ||
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
           (static_cast<bool>(LIM) && (mask_LIM == nullptr)) ||
           B_norm == nullptr ||
           Bstar_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using Operator = koliop::collision::Operator;

        auto* const an_operator = new Operator{static_cast<uint64_t>(the_mu_extent),
                                               static_cast<uint64_t>(the_vpar_extent),
                                               static_cast<uint64_t>(the_r_extent),
                                               static_cast<uint64_t>(the_theta_extent),
                                               static_cast<uint64_t>(the_phi_extent),
                                               static_cast<uint64_t>(the_species_extent),
                                               static_cast<uint64_t>(the_global_domain_r_extent),
                                               static_cast<uint64_t>(the_global_domain_theta_extent),
                                               static_cast<uint64_t>(the_local_domain_r_offset),
                                               static_cast<uint64_t>(the_local_domain_theta_offset),
                                               R0,
                                               static_cast<bool>(collision_interspecies),
                                               static_cast<bool>(SOL),
                                               static_cast<bool>(LIM),
                                               static_cast<uint64_t>(ir_SOL_separatrix),
                                               hat_As,
                                               hat_Zs,
                                               mug,
                                               vparg,
                                               coeff_intdmu,
                                               coeff_intdvpar,
                                               rg,
                                               nustar0_r,
                                               q_r,
                                               mask_buffer_r,
                                               mask_LIM,
                                               B_norm,
                                               Bstar_s};

        KOLIOP_ASSERT(an_operator != nullptr);

        *an_operator_handle = reinterpret_cast<koliop_Operator>(an_operator);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Release(::koliop_Operator an_operator) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        ::koliop_Fence(an_operator);

        // Freeing a nullptr is *fine*.
        delete reinterpret_cast<koliop::collision::Operator*>(an_operator);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Fence(::koliop_Operator an_operator) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        // NOTE: Maybe we should offer to the user the possibility of choosing
        // which stream to use for computation and for fencing.
        reinterpret_cast<const koliop::collision::Operator*>(an_operator)->FenceAll();

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMoments(::koliop_Operator an_operator,
                                      double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->ComputeMoments(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_PrecomputeNu(::koliop_Operator an_operator) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->PrecomputeNu();

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMaxwellianEvolution(::koliop_Operator an_operator,
                                                  double            dt,
                                                  double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->ComputeMaxwellianEvolution(static_cast<koliop::ComputeFloatType>(dt), f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeQparOnNT(::koliop_Operator an_operator,
                                       const double*     f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->ComputeQparOnNT(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeCpar(::koliop_Operator an_operator,
                                   double            dt,
                                   double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<const koliop::collision::Operator*>(an_operator)->ComputeCpar(static_cast<koliop::ComputeFloatType>(dt), f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeMaxwellianUnshiftedLoc(::koliop_Operator an_operator,
                                                     double*           f5d_s,
                                                     int32_t           choice) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr ||
           !(choice == 3 || choice == 4)) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<const koliop::collision::Operator*>(an_operator)->ComputeMaxwellianUnshiftedLoc(static_cast<uint32_t>(choice), f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Apply(::koliop_Operator an_operator,
                             double            dt,
                             double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->Apply(static_cast<koliop::ComputeFloatType>(dt), f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeDensityCorrection(::koliop_Operator an_operator,
                                                double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<const koliop::collision::Operator*>(an_operator)->ComputeDensityCorrection(f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ComputeUparTCorrection(::koliop_Operator an_operator,
                                              double            dt,
                                              double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->ComputeUparTCorrection(static_cast<koliop::ComputeFloatType>(dt), f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_ExtractRaw(::koliop_Operator an_operator,
                                  int32_t           the_value_identifier,
                                  void*             an_output_buffer) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           the_value_identifier < 0 ||
           an_output_buffer == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->ExtractRaw(static_cast<uint32_t>(the_value_identifier), an_output_buffer);

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Collision(::koliop_Operator an_operator,
                                 double            dt,
                                 double*           f5d_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           f5d_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->Collision(static_cast<koliop::ComputeFloatType>(dt), f5d_s);

        return KOLIOP_STATUS_SUCCESS;
    });
}
