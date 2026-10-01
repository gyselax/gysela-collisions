// SPDX-License-Identifier: MIT

#include "KOLIOP/koliop.h"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

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
#if defined(KOLIOP_BACKEND_IS_GPU_LIKE)
            std::cout << "  KOLIOP_BACKEND_IS_GPU_LIKE: true\n";
#else
            std::cout << "  KOLIOP_BACKEND_IS_GPU_LIKE: false\n";
#endif
#if defined(KOLIOP_BACKEND_IS_CPU_LIKE)
            std::cout << "  KOLIOP_BACKEND_IS_CPU_LIKE: true\n";
#else
            std::cout << "  KOLIOP_BACKEND_IS_CPU_LIKE: false\n";
#endif
#if defined(KOLIOP_ASSERT_ENABLED)
            std::cout << "  KOLIOP_ASSERT_ENABLED: true\n";
#else
            std::cout << "  KOLIOP_ASSERT_ENABLED: false\n";
#endif
            std::cout << "  Expecting f5d_s as (mu,vpar),(theta,r,phi),species\n";
            std::cout << "  Expecting input buffers on device\n";
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
            std::cout << "  KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: true\n";
#else
            std::cout << "  KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS: false\n";
#endif
#if defined(KOLIOP_CONSTANT_MEMORY_USAGE)
            std::cout << "  KOLIOP_CONSTANT_MEMORY_USAGE: true\n";
#else
            std::cout << "  KOLIOP_CONSTANT_MEMORY_USAGE: false\n";
#endif

            std::cout << "  koliop::collision::Constant::kDisjointExecutionSpaceCount: " << koliop::collision::Constant::kDisjointExecutionSpaceCount << "\n";
            std::cout << "  koliop::collision::Constant::kDiscardedStreamCount: " << koliop::collision::Constant::kDiscardedStreamCount << "\n";
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
            auto the_configuration = Kokkos::InitializationSettings()
                                         .set_device_id(static_cast<int>(a_device_index))
                                         .set_disable_warnings(!static_cast<bool>(with_verbose_initialization));

            // If KOLIOP_SET_NUM_THREADS is set, use that value.
            auto* koliop_set_num_threads_string = std::getenv("KOLIOP_SET_NUM_THREADS");
            if(koliop_set_num_threads_string) {
                a_host_thread_count = std::stoi(koliop_set_num_threads_string);
            }

            if(a_host_thread_count > 0) {
                the_configuration.set_num_threads(a_host_thread_count);
            }

            Kokkos::initialize(the_configuration);
        }

        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(static_cast<bool>(with_verbose_initialization)) {
            Kokkos::print_configuration(std::cout, true);
        }

        std::cout << std::flush;

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Deinitialize(int8_t do_kokkos_deinitialization) {
    return koliop::error::FromException([&]() {
        if(static_cast<bool>(do_kokkos_deinitialization)) {
            if(!Kokkos::is_initialized()) {
                return KOLIOP_STATUS_DEINITIALIZATION_FAILURE;
            }

            Kokkos::finalize();

            if(Kokkos::is_initialized()) {
                return KOLIOP_STATUS_DEINITIALIZATION_FAILURE;
            }
        }

        return KOLIOP_STATUS_SUCCESS;
    });
}

::koliop_Status koliop_Create(::koliop_Operator* an_operator_handle,
                              size_t             mu_extent,
                              size_t             vpar_extent,
                              size_t             tor1_extent,
                              size_t             tor2_extent,
                              size_t             tor3_extent,
                              size_t             species_extent,
                              size_t             global_domain_r_extent,
                              size_t             global_domain_theta_extent,
                              size_t             local_domain_r_offset,
                              size_t             local_domain_theta_offset,
                              int8_t             collision_interspecies,
                              int8_t             SOL,
                              int8_t             LIM,
                              int64_t            ir_SOL_separatrix,
                              const double*      hat_As,
                              const double*      hat_Zs,
                              const double*      mug,
                              const double*      vparg,
                              const double*      coeff_intdmu,
                              const double*      coeff_intdvpar,
                              const double*      coeff_AD_r,
                              const double*      mask_buffer_r,
                              const double*      mask_LIM,
                              const double*      B_norm,
                              const double*      Bstar_s) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(std::getenv("KOLIOP_DEBUG_ALL") != nullptr ||
           std::getenv("KOLIOP_DEBUG_koliop_Create") != nullptr) {
            std::cout << "[KOLIOP][DEBUG] !(0 <= ir_SOL_separatrix && static_cast<size_t>(ir_SOL_separatrix) <= global_domain_r_extent) : "
                      << (!(0 <= ir_SOL_separatrix && static_cast<size_t>(ir_SOL_separatrix) <= global_domain_r_extent)) << "\n";
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
            // Assert for powers of t
            std::cout << "[KOLIOP][DEBUG] ((mu_extent - 1) & mu_extent) != 0 : " << (((mu_extent - 1) & mu_extent) != 0) << "\n";
            std::cout << "[KOLIOP][DEBUG] ((vpar_extent - 1) & vpar_extent) != 0 : " << (((vpar_extent - 1) & vpar_extent) != 0) << "\n";
#endif
            std::cout << "[KOLIOP][DEBUG] hat_As == nullptr : " << (hat_As == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] hat_Zs == nullptr : " << (hat_Zs == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] mug == nullptr : " << (mug == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] vparg == nullptr : " << (vparg == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] coeff_intdmu == nullptr : " << (coeff_intdmu == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] coeff_intdvpar == nullptr : " << (coeff_intdvpar == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] coeff_AD_r == nullptr : " << (coeff_AD_r == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] mask_buffer_r == nullptr : " << (mask_buffer_r == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] (static_cast<bool>(LIM) && (mask_LIM == nullptr)) : " << ((static_cast<bool>(LIM) && (mask_LIM == nullptr))) << "\n";
            std::cout << "[KOLIOP][DEBUG] B_norm == nullptr : " << (B_norm == nullptr) << "\n";
            std::cout << "[KOLIOP][DEBUG] Bstar_s == nullptr : " << (Bstar_s == nullptr) << "\n";
            std::cout << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(mu_extent) : " << static_cast<uint64_t>(mu_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(vpar_extent) : " << static_cast<uint64_t>(vpar_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(tor1_extent) : " << static_cast<uint64_t>(tor1_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(tor2_extent) : " << static_cast<uint64_t>(tor2_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(tor3_extent) : " << static_cast<uint64_t>(tor3_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(species_extent) : " << static_cast<uint64_t>(species_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(global_domain_r_extent) : " << static_cast<uint64_t>(global_domain_r_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(global_domain_theta_extent) : " << static_cast<uint64_t>(global_domain_theta_extent) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(local_domain_r_offset) : " << static_cast<uint64_t>(local_domain_r_offset) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(local_domain_theta_offset) : " << static_cast<uint64_t>(local_domain_theta_offset) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<bool>(collision_interspecies) : " << static_cast<bool>(collision_interspecies) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<bool>(SOL) : " << static_cast<bool>(SOL) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<bool>(LIM) : " << static_cast<bool>(LIM) << "\n";
            std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(ir_SOL_separatrix) : " << static_cast<uint64_t>(ir_SOL_separatrix) << "\n";
            std::cout << "[KOLIOP][DEBUG] hat_As : " << hat_As << "\n";
            std::cout << "[KOLIOP][DEBUG] hat_Zs : " << hat_Zs << "\n";
            std::cout << "[KOLIOP][DEBUG] mug : " << mug << "\n";
            std::cout << "[KOLIOP][DEBUG] vparg : " << vparg << "\n";
            std::cout << "[KOLIOP][DEBUG] coeff_intdmu : " << coeff_intdmu << "\n";
            std::cout << "[KOLIOP][DEBUG] coeff_intdvpar : " << coeff_intdvpar << "\n";
            std::cout << "[KOLIOP][DEBUG] coeff_AD_r : " << coeff_AD_r << "\n";
            std::cout << "[KOLIOP][DEBUG] mask_buffer_r : " << mask_buffer_r << "\n";
            std::cout << "[KOLIOP][DEBUG] mask_LIM : " << mask_LIM << "\n";
            std::cout << "[KOLIOP][DEBUG] B_norm : " << B_norm << "\n";
            std::cout << "[KOLIOP][DEBUG] Bstar_s : " << Bstar_s << "\n";

            std::cout << std::flush;
        }

        if(an_operator_handle == nullptr ||
           !(0 <= ir_SOL_separatrix && static_cast<size_t>(ir_SOL_separatrix) <= global_domain_r_extent) ||
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
           // Assert for powers of two.
           ((mu_extent - 1) & mu_extent) != 0 ||
           ((vpar_extent - 1) & vpar_extent) != 0 ||
#endif
           hat_As == nullptr ||
           hat_Zs == nullptr ||
           mug == nullptr ||
           vparg == nullptr ||
           coeff_intdmu == nullptr ||
           coeff_intdvpar == nullptr ||
           coeff_AD_r == nullptr ||
           mask_buffer_r == nullptr ||
           (static_cast<bool>(LIM) && (mask_LIM == nullptr)) ||
           B_norm == nullptr ||
           Bstar_s == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        using Operator = koliop::collision::Operator;

        auto* const an_operator = new Operator{static_cast<uint64_t>(mu_extent),
                                               static_cast<uint64_t>(vpar_extent),
                                               static_cast<uint64_t>(tor1_extent),
                                               static_cast<uint64_t>(tor2_extent),
                                               static_cast<uint64_t>(tor3_extent),
                                               static_cast<uint64_t>(species_extent),
                                               static_cast<uint64_t>(global_domain_r_extent),
                                               static_cast<uint64_t>(global_domain_theta_extent),
                                               static_cast<uint64_t>(local_domain_r_offset),
                                               static_cast<uint64_t>(local_domain_theta_offset),
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
                                               coeff_AD_r,
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
                                  int32_t           value_identifier,
                                  void*             an_output_buffer) {
    return koliop::error::FromException([&]() {
        if(!Kokkos::is_initialized()) {
            return KOLIOP_STATUS_INITIALIZATION_FAILURE;
        }

        if(an_operator == nullptr ||
           value_identifier < 0 ||
           an_output_buffer == nullptr) {
            return KOLIOP_STATUS_INVALID_VALUE;
        }

        reinterpret_cast<koliop::collision::Operator*>(an_operator)->ExtractRaw(static_cast<uint32_t>(value_identifier), an_output_buffer);

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
