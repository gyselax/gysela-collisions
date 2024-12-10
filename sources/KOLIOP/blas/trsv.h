// SPDX-License-Identifier: MIT

#ifndef KOLIOP_BLAS_TRSV_H
#define KOLIOP_BLAS_TRSV_H

#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>
#include <KokkosBlas2_gemv.hpp>

#include "KOLIOP/blas/kind.h"

namespace koliop {
    namespace blas {

        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRSV definition
        ////////////////////////////////////////////////////////////////////////

        /// See BatchedBlockTridiagonalByBlockTRSV for a detailed matrix
        /// layout explanation.
        /// @tparam BatchTraits
        /// @tparam kSmallBlockExtent
        template <typename BatchTraits,
                  // TODO(Etienne M): See if kFlattenedBlockExtent which is
                  // constexpr, can help improve the performance when used as an
                  // template argument. kBigBlockExtent = kFlattenedBlockExtent.
                  // This would remove A_matrix_extent argument.
                  OrdinalType kSmallBlockExtent>
        class BatchedBlockTridiagonalByBlockTRSV {
        public:
            using BatchTraitsType = BatchTraits;

            using value_type = typename BatchTraitsType::value_type;

            using ExecutionSpaceType = typename BatchTraitsType::ExecutionSpaceType;

            using VectorKindType = kind::vector::BatchedBlockByBlock<BatchTraitsType,
                                                                     kSmallBlockExtent>;

            using MatrixKindType = kind::matrix::BatchedBlockTridiagonalByBlock<BatchTraitsType,
                                                                                kSmallBlockExtent>;

        protected:
            // using ConcurrencyPolicy = KokkosBatched::Mode::Serial;
            using CachingPolicy = KokkosBatched::Algo::Level2::Unblocked;

            using TRSVLower = KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Lower,
                                                        KokkosBatched::Trans::NoTranspose,
                                                        KokkosBatched::Diag::Unit,
                                                        CachingPolicy>;

            using TRSVUpper = KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Upper,
                                                        KokkosBatched::Trans::NoTranspose,
                                                        KokkosBatched::Diag::NonUnit,
                                                        CachingPolicy>;

            /// NOTE: Why
            /// GEMM<NoTranspose, NoTranspose, Level3::Unblocked>::invoke is
            /// allowed to take the same arguments than
            /// GEMV<NoTranspose, Level2::Unblocked>::invoke ?
            using GEMV = KokkosBlas::SerialGemv<KokkosBatched::Trans::NoTranspose,
                                                CachingPolicy>;

        public:
            /// TRSV, solve Ax = b where A is a LU factorization output
            /// by BatchedBlockTridiagonalByBlockTRF::Factorize().
            /// @param rhs In place resolution. rhs is overwritten.
            void Solve(ExecutionSpaceType an_execution_space,
                       const value_type*  A_matrix,
                       OrdinalType        A_matrix_extent,
                       OrdinalType        M_matrix_extent,
                       value_type*        rhs) const;

        protected:
        };


        ////////////////////////////////////////////////////////////////////////
        // BatchedBlockTridiagonalByBlockTRSV methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename BatchTraits,
                  OrdinalType kSmallBlockExtent>
        void BatchedBlockTridiagonalByBlockTRSV<BatchTraits,
                                                kSmallBlockExtent>::Solve(ExecutionSpaceType an_execution_space,
                                                                          const value_type*  A_matrix,
                                                                          OrdinalType        A_matrix_extent,
                                                                          OrdinalType        M_matrix_extent,
                                                                          value_type*        rhs) const {

            KOLIOP_ASSERT(A_matrix != nullptr);
            KOLIOP_ASSERT(rhs != nullptr);
            // A_matrix_extent must be divisible by
            // BatchTraitsType::kBatchSize. Else we need padding.
            KOLIOP_ASSERT(A_matrix_extent % BatchTraitsType::kBatchSize == 0);
            KOLIOP_ASSERT(M_matrix_extent > 0);

            const OrdinalType kBatchCount     = BatchTraitsType::BatchCount(A_matrix_extent);
            const OrdinalType kBigBlockExtent = M_matrix_extent;

            static_assert(MatrixKindType::kDiagonalCount == 3, "");

            typename MatrixKindType::ConstVectorType A_matrix_view{reinterpret_cast<const typename BatchTraitsType::ThreadVectorType*>(A_matrix),
                                                                   static_cast<std::size_t>(kBatchCount),
                                                                   static_cast<std::size_t>(kBigBlockExtent)};

            typename VectorKindType::VectorType rhs_vector_view{reinterpret_cast<typename BatchTraitsType::ThreadVectorType*>(rhs),
                                                                static_cast<std::size_t>(kBatchCount),
                                                                static_cast<std::size_t>(kBigBlockExtent)};

            const auto kernel = KOKKOS_LAMBDA(OrdinalType batch_element_index, OrdinalType batch_index) {

                auto matrix_batch_element = Kokkos::subview(A_matrix_view,
                                                            batch_index,
                                                            // TODO(Etienne M): These ALL() are constant, should we
                                                            // hardcode them to facilitate the compiler's work ?
                                                            Kokkos::ALL(), // kBigBlockExtent
                                                            Kokkos::ALL(), // kDiagonalCount
                                                            Kokkos::ALL(), // kSmallBlockExtent
                                                            Kokkos::ALL(), // kSmallBlockExtent
                                                            batch_element_index);

                auto vector_batch_element = Kokkos::subview(rhs_vector_view,
                                                            batch_index,
                                                            // TODO(Etienne M): These ALL() are constant, should we
                                                            // hardcode them to facilitate the compiler's work ?
                                                            Kokkos::ALL(), // kBigBlockExtent
                                                            Kokkos::ALL(), // kSmallBlockExtent
                                                            batch_element_index);

                // Solves Mx = vec, overwriting vec with x.

                // M =
                // B_0 C_0       0
                // A_1 B_1 C_1
                //     A_. A_. B_.
                // 0       A_n B_n

                // rhs =
                // b_0
                // b_1
                // b_.
                // b_n

                // B_n = L x U
                // With kSmallBlockExtent == 3
                // L =
                // 1   0
                // a 1
                // b c 1

                // U =
                // a b c
                //   d e
                // 0   f

                // b_n =
                // a
                // b
                // c

                // Forward substitution
                {
                    // Applying the naming above, we have:
                    auto B      = Kokkos::subview(matrix_batch_element, 0, 1, Kokkos::ALL(), Kokkos::ALL());
                    auto A_next = Kokkos::subview(matrix_batch_element, 0 /* Putting 1 here is illegal ? */, 0, Kokkos::ALL(), Kokkos::ALL());

                    auto b      = Kokkos::subview(vector_batch_element, 0, Kokkos::ALL());
                    auto b_next = Kokkos::subview(vector_batch_element, 0 /* Putting 1 here is illegal ? */, Kokkos::ALL());

                    for(OrdinalType small_block_index = 0;
                        small_block_index < (kBigBlockExtent - 1);
                        ++small_block_index) {

                        // TODO(Etienne M): Recreate view or assign ? Which is
                        // faster ?

                        B.assign_data(&matrix_batch_element(small_block_index, 1, 0, 0));

                        TRSVLower::invoke(static_cast<value_type>(1.0), B, b); // Solve: B_lower * y = b, y is put into b

                        A_next.assign_data(&matrix_batch_element(small_block_index + 1, 0, 0, 0));
                        b_next.assign_data(&vector_batch_element(small_block_index + 1, 0));

                        GEMV::invoke(static_cast<value_type>(-1.0), A_next, b,
                                     static_cast<value_type>(1.0), b_next); // b_next = 1.0 * b_next + -1.0 * A_next * b

                        // TODO(Etienne M): Do an assign_data for b or copy ? which is faster ?
                        b = b_next;
                    }

                    B.assign_data(&matrix_batch_element(kBigBlockExtent - 1, 1, 0, 0));

                    TRSVLower::invoke(static_cast<value_type>(1.0), B, b);
                }

                // Backward substitution
                {
                    // Applying the naming above, we have:
                    auto B      = Kokkos::subview(matrix_batch_element, 0, 1, Kokkos::ALL(), Kokkos::ALL());
                    auto C_next = Kokkos::subview(matrix_batch_element, 0, 0, Kokkos::ALL(), Kokkos::ALL());

                    auto b      = Kokkos::subview(vector_batch_element, kBigBlockExtent - 1, Kokkos::ALL());
                    auto b_next = Kokkos::subview(vector_batch_element, 0, Kokkos::ALL());

                    // NOTE: when iterating in reverse order, the *next* element
                    // is the *previous* one when iterating in natural order.

                    for(OrdinalType reversed_small_block_index = kBigBlockExtent;
                        reversed_small_block_index > 1;
                        --reversed_small_block_index) {

                        const OrdinalType small_block_index = reversed_small_block_index - 1;

                        // TODO(Etienne M): Recreate view or assign ? Which is
                        // faster ?

                        B.assign_data(&matrix_batch_element(small_block_index, 1, 0, 0));

                        TRSVUpper::invoke(static_cast<value_type>(1.0), B, b); // Solve: B_upper * x = b, x is put into b

                        C_next.assign_data(&matrix_batch_element(small_block_index - 1, 2, 0, 0));
                        b_next.assign_data(&vector_batch_element(small_block_index - 1, 0));

                        GEMV::invoke(static_cast<value_type>(-1.0), C_next, b,
                                     static_cast<value_type>(1.0), b_next); // b_next = 1.0 * b_next + -1.0 * C_next * b

                        b = b_next;
                    }

                    B.assign_data(&matrix_batch_element(0, 1, 0, 0));

                    TRSVUpper::invoke(static_cast<value_type>(1.0), B, b);
                }
            };

            const auto a_policy = Kokkos::MDRangePolicy<
                // Kokkos::LaunchBounds<64, 1>, Kokkos::IndexType<OrdinalType>,
                ExecutionSpaceType,
                Kokkos::Rank<2,
                             // TODO(Etienne M): Tune blocking.
                             Kokkos::Iterate::Left,
                             Kokkos::Iterate::Left>>{an_execution_space,
                                                     {0, 0},
                                                     {BatchTraitsType::kVectorPerBatch, kBatchCount}};

            Kokkos::parallel_for("BatchedBlockTridiagonalByBlockTRSV::Solve",
                                 a_policy, kernel);
        }

    } // namespace blas
} // namespace koliop

// #include <Kokkos_Random.hpp>

// #include "KOLIOP/utility/copy.h"

// namespace koliop {
//     namespace blas {
//         namespace detail {
//             template <typename ExecutionSpaceType,
//                       typename Factorizer,
//                       typename Solver,
//                       typename MV,
//                       typename View6D,
//                       typename View4D>
//             void DoResidualTest(ExecutionSpaceType an_execution_space,
//                                 Factorizer&&       factorizer,
//                                 Solver&&           solver,
//                                 MV&&               mv,
//                                 View6D&&           matrix_buffer,
//                                 View4D&&           original_x_buffer,
//                                 View4D&&           computed_x_buffer) {

//                 using value_type = typename std::remove_reference_t<View6D>::value_type;

//                 std::cout << "Input  | Batch count: " << matrix_buffer.extent(0)
//                           << " | Small block count: " << matrix_buffer.extent(1)
//                           << " | Batch size: " << matrix_buffer.extent(5) << "\n";

//                 Kokkos::resize(matrix_buffer,
//                                32 * 1024, // 1, 2, 3, 4, 8, 13
//                                256,
//                                3,
//                                3,
//                                3,
//                                matrix_buffer.extent(5));
//                 Kokkos::resize(original_x_buffer,
//                                matrix_buffer.extent(0),
//                                matrix_buffer.extent(1),
//                                matrix_buffer.extent(4),
//                                matrix_buffer.extent(5));
//                 Kokkos::resize(computed_x_buffer,
//                                matrix_buffer.extent(0),
//                                matrix_buffer.extent(1),
//                                matrix_buffer.extent(4),
//                                matrix_buffer.extent(5));

//                 std::cout << "Actual | Batch count: " << matrix_buffer.extent(0)
//                           << " | Small block count: " << matrix_buffer.extent(1)
//                           << " | Batch size: " << matrix_buffer.extent(5) << "\n";

//                 // NOTE: We do not guarantee that we have a weakly diagonally
//                 // dominant matrix !!
//                 // - Random [0, 1.0) matrix_buffer
//                 // - Random [0, 1.0) original_x_buffer

//                 Kokkos::deep_copy(original_x_buffer, 3.14159265358979323846);

//                 const auto DoFillMatrix = [&]() {
//                     const auto a_matrix_policy = Kokkos::MDRangePolicy<Kokkos::Rank<6,
//                                                                                     Kokkos::Iterate::Left,
//                                                                                     Kokkos::Iterate::Left>>{{0, 0, 0, 0, 0, 0},
//                                                                                                             {matrix_buffer.extent(0),
//                                                                                                              matrix_buffer.extent(1),
//                                                                                                              matrix_buffer.extent(2),
//                                                                                                              matrix_buffer.extent(3),
//                                                                                                              matrix_buffer.extent(4),
//                                                                                                              matrix_buffer.extent(5)}};

//                     // Kokkos::parallel_for(
//                     //     "DoResidualTest::RandomFilling",
//                     //     a_matrix_policy,
//                     //     KOKKOS_LAMBDA(auto a, auto b, auto c, auto d, auto e, auto f) {
//                     //         if(d == e && c == 1) {
//                     //             matrix_buffer(a, b, c, d, e, f) = 10.0;
//                     //         } else {
//                     //             matrix_buffer(a, b, c, d, e, f) = 2.0;
//                     //         }
//                     //     });

//                     // NOTE: that when factorizing Flo32 matrices, we may have
//                     // NANs because no pivoting is done.
//                     Kokkos::Random_XorShift64_Pool<> random_pool(0xDEADBEEF);
//                     Kokkos::fill_random(matrix_buffer, random_pool, 0.0, 1.0);
//                 };

//                 DoFillMatrix();

//                 // - matrix_buffer * original_x_buffer = computed_x_buffer

//                 mv.Apply(an_execution_space,
//                              1.0,
//                              matrix_buffer,
//                              matrix_buffer.extent(0) * matrix_buffer.extent(5), matrix_buffer.extent(1),
//                              original_x_buffer.data(),
//                              0.0,
//                              computed_x_buffer.data());

//                 koliop::utility::Copy::Deep(an_execution_space,
//                                             computed_x_buffer,
//                                             original_x_buffer);

//                 // - Solver matrix_buffer * y = computed_x_buffer, y put into computed_x_buffer

//                 factorizer.Factorize(an_execution_space,
//                                          matrix_buffer.data(),
//                                          matrix_buffer.extent(0) * matrix_buffer.extent(5), matrix_buffer.extent(1));

//                 solver.Solve(an_execution_space,
//                                  matrix_buffer.data(),
//                                  matrix_buffer.extent(0) * matrix_buffer.extent(5), matrix_buffer.extent(1),
//                                  computed_x_buffer.data());

//                 // - Compare computed_x_buffer with original_x_buffer

//                 DoFillMatrix();

//                 // b = b - Ax, b must equals zero.
//                 mv.Apply(an_execution_space,
//                              1.0,
//                              matrix_buffer,
//                              matrix_buffer.extent(0) * matrix_buffer.extent(5), matrix_buffer.extent(1),
//                              computed_x_buffer.data(),
//                              -1.0,
//                              original_x_buffer.data());

//                 const auto a_vector_policy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>{{0, 0, 0, 0},
//                                                                                     {matrix_buffer.extent(0),
//                                                                                      matrix_buffer.extent(1),
//                                                                                      matrix_buffer.extent(3),
//                                                                                      matrix_buffer.extent(5)}};

//                 value_type residual_sum;

//                 Kokkos::parallel_reduce(
//                     "DoResidualTest::ResidualComputation",
//                     a_vector_policy,
//                     KOKKOS_LAMBDA(const int& a, const int& b, const int& c, const int& d,
//                                   // NOTE: we need const int& here, that weird..
//                                   value_type& sum) {
//                         // NOTE: There is no guarantee that we have at least two
//                         // value per SIMD Vector
//                         if(Kokkos::isnan(original_x_buffer(a, b, c, d))) {
//                             printf("%d %d %d %d\n", a, b, c, d);
//                         }

//                         sum += original_x_buffer(a, b, c, d) *
//                                    original_x_buffer(a, b, c, d);
//                     },
//                     residual_sum);

//                 std::cout << "L2 of b (residual):"
//                           << Kokkos::sqrt(residual_sum)
//                           << " (" << residual_sum << ")\n";
//             }
//         } // namespace detail
//     }     // namespace blas
// } // namespace koliop

#endif
