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
            using value_type         = Value;
            using ExecutionSpaceType = ExecutionSpace;

            /// We wish to saturate the hardware.
            /// - On CPU this means using SIMD instructions.
            /// - On GPU this means using (intra lane) SIMD such as double2
            /// combined with wavefront coalesced accesses. On AMD GPUs, a
            /// wavefront is 64 threads but we could go even lower and only seek
            /// to do 16 thread coalesced accesses as wavefronts are processed
            /// by groups of 16 threads.
            /// kBatchSize represents the length of an efficient load for a
            /// given SIMD architecture. On AVX2, this would be a 32 bytes (256
            /// bits, a whole ymm register equivalent to half an x86 cache line)
            /// and on a GPU this could be 128 bytes. So for a batch of double:
            /// CPU  (AVX2):  32 bytes / sizeof(double) -> kBatchSize =  4
            /// GPU (MI250): 128 bytes / sizeof(double) -> kBatchSize = 16
            /// Now that we know how much we have to load at once to be
            /// efficient,  we need to figure out, how many bytes one thread can
            /// load, this is the role of kVectorSize.
            static inline constexpr OrdinalType kBatchSize = KokkosBatched::DefaultVectorLength<value_type,
                                                                                                typename ExecutionSpaceType::memory_space>::value;

            /// - On CPU, the efficient load size is given by the cache line but
            /// ultimately, by the largest SIMD register's width. Each thread
            /// can perform one of such load so kBatchSize == kVectorSize.
            /// - On GPU it shall represent an inter lane SIMD such as double4.
            /// Indeed, one can very much see the GPU thread's state as
            /// contained in a SIMD register's lane. On GPU though, there is the
            /// concept of *interlane* SIMD which comes in the form of
            /// float2, double4 etc. As of 2023, doubleN is not really a SIMD
            /// registers and serve more to do efficient loads. In the case of
            /// a GPU, kVectorSize may be == 4 regardless of kBatchSize.
            ///
            /// NOTE: While batching is required to produce efficient memory
            /// access on GPUs, using single double instead of double4/double2
            /// exposes more concurrency. We are a bit faster, on MI250X, when
            /// using kVectorSize = 1, instead of the default which is 4.
            ///
            static inline constexpr OrdinalType kVectorSize = KokkosBatched::DefaultInternalVectorLength<value_type,
                                                                                                         typename ExecutionSpaceType::memory_space>::value;

            /// On AMD GPUs,    its 16 / 4 (double4) = 4
            //     Nvidia GPUs, its  8 / 2 (double2) = 4
            ///    AVX256,      its  4 / 4 (double)  = 1
            static inline constexpr OrdinalType kVectorPerBatch = kBatchSize / kVectorSize;

            using ThreadVectorType = KokkosBatched::Vector<KokkosBatched::SIMD<value_type>,
                                                           kVectorSize>;

        public:
            KOKKOS_INLINE_FUNCTION static constexpr OrdinalType
            BatchCount(OrdinalType an_extent);
        };


        ////////////////////////////////////////////////////////////////////////
        // FakeBatchTraits definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Value,
                  typename ExecutionSpace>
        struct FakeBatchTraits : public BatchTraits<Value, ExecutionSpace> {
        public:
            using BaseType = BatchTraits<Value, ExecutionSpace>;

            static inline constexpr OrdinalType kBatchSize = 1;

            static inline constexpr OrdinalType kVectorSize = 1;

            static inline constexpr OrdinalType kVectorPerBatch = kBatchSize / kVectorSize;

            using ThreadVectorType = typename BaseType::value_type;

        public:
            KOKKOS_INLINE_FUNCTION static constexpr OrdinalType
            BatchCount(OrdinalType an_extent);
        };


        ////////////////////////////////////////////////////////////////////////
        // BatchTraits methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Value,
                  typename ExecutionSpace>
        KOKKOS_INLINE_FUNCTION constexpr OrdinalType
        BatchTraits<Value, ExecutionSpace>::BatchCount(OrdinalType an_extent) {
            return an_extent / kBatchSize;
        }


        ////////////////////////////////////////////////////////////////////////
        // FakeBatchTraits methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Value,
                  typename ExecutionSpace>
        KOKKOS_INLINE_FUNCTION constexpr OrdinalType
        FakeBatchTraits<Value, ExecutionSpace>::BatchCount(OrdinalType an_extent) {
            return an_extent / kBatchSize;
        }

    } // namespace blas
} // namespace koliop

#endif
