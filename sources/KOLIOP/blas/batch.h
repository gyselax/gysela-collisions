// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_BATCH_H
#define KOLIOP_BLAS_BATCH_H

#include <KokkosBatched_Vector.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // BatchTraits definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Value,
                  typename ExecutionSpace>
        struct BatchTraits {
        public:
            using ValueType          = Value;
            using ExecutionSpaceType = ExecutionSpace;

            /// We wish to saturate the hardware.
            /// - On CPU this means using SIMD instructions.
            /// - On GPU this means using (intra lane) SIMD such as double2
            /// combined with wavefront coalesced accesses. On AMD GPUs, a
            /// wavefront is 64 but we could go even lower and only seek to do
            /// 16 thread coalesced accesses as wave fronts a processed by
            /// groups of 16 threads.
            /// kBatchSize represents the vector length from the hardware POV.
            /// The vector length from the thread POV may differ due to intra
            /// lane SIMD such as double2.
            static inline constexpr OrdinalType kBatchSize = KokkosBatched::DefaultVectorLength<ValueType,
                                                                                                typename ExecutionSpaceType::memory_space>::value;

            /// We define the vector type from the thread POV.
            /// - On CPU it is similar to the vector type from the hardware POV
            /// say, a ymm register.
            /// - On GPU it shall represent an inter lane SIMD such as double4.
            static inline constexpr OrdinalType kVectorSize = KokkosBatched::DefaultInternalVectorLength<ValueType,
                                                                                                         typename ExecutionSpaceType::memory_space>::value;

            static inline constexpr OrdinalType kVectorPerBatch = kBatchSize / kVectorSize;

            using ThreadVectorType = KokkosBatched::Vector<KokkosBatched::SIMD<ValueType>,
                                                           kVectorSize>;
        };

    } // namespace blas
} // namespace koliop

#endif
