// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_DEFECT_H
#define KOLIOP_UTILITY_DEFECT_H

#include <Kokkos_Core.hpp>

namespace koliop {
    namespace utility {

        /// Mostly used to select a team size which could be an integer or a
        /// Kokkos::AUTO_t.
        template <typename ExecutionSpace,
                  typename T0,
                  typename T1>
        constexpr auto
        Select(T0&& the_openmp_value,
               T1&& the_hip_value) {

            // NOTE: We need these if def shenanigans to cover an other defect.
            // The fact that Kokkos::HIP simply does not exist if the backend is
            // not enabled. It could at least exist and be empty.. for
            // meta-programming' sake. That is a defect IMO.

#if defined(KOKKOS_ENABLE_OPENMP)
            using OpenMPExecutionSpace = Kokkos::OpenMP;
#else
            // Unique type name.
            using OpenMPExecutionSpace = struct DummyOpenMP {};
#endif

#if defined(KOKKOS_ENABLE_HIP)
            using HIPExecutionSpace = Kokkos::HIP;
#else
            // Unique type name.
            using HIPExecutionSpace = struct DummyHIP {};
#endif

#if defined(KOKKOS_ENABLE_CUDA)
            using CUDAExecutionSpace = Kokkos::Cuda;
#else
            // Unique type name.
            using CUDAExecutionSpace = struct DummyCUDA {};
#endif

            if constexpr(std::is_same_v<ExecutionSpace, OpenMPExecutionSpace>) {
                return the_openmp_value;
            } else if constexpr(std::is_same_v<ExecutionSpace, HIPExecutionSpace>) {
                return the_hip_value;
            } else if constexpr(std::is_same_v<ExecutionSpace, CUDAExecutionSpace>) {
                return the_hip_value;
            } else {
                // NOTE: A shame there is no *else constexpr*.
                static_assert(std::is_same_v<ExecutionSpace, OpenMPExecutionSpace> ||
                                   std::is_same_v<ExecutionSpace, HIPExecutionSpace>||
                                  std::is_same_v<ExecutionSpace, CUDAExecutionSpace> ,
                              "Execution space not supported we need to work on that.");
            }
        }

    } // namespace utility
} // namespace koliop

#endif
