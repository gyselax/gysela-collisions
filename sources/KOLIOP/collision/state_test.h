// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // State definition
        ////////////////////////////////////////////////////////////////////////


        /// Execution Spaces:
        ///     Where you physically execute
        ///     Memory Spaces:
        ///         Where you physically store
        ///         Memory Layout:
        ///             How to index
        ///         Memory Traits:
        ///             Modifier: stream, random, atomic, texture memory
        /// Execution Patterns:
        ///     Parallel for/scan/reduce
        /// Execution Policies:
        ///     Range, team, dynamic or static scheduling
        ///
        template <typename Policy>
        class State {
        public:
            using PolicyType = Policy;

            template <typename T>
            using MatrixType = Kokkos::View<T**,
                                            Kokkos::DefaultExecutionSpace::memory_space>;
            template <typename T>
            using VectorType = Kokkos::View<T*,
                                            Kokkos::DefaultExecutionSpace::memory_space>;

        public:
            State(std::size_t a_width)
                : the_data_compute_queue_{Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace{}, 1)[0]}
                , the_matrix_A_{Kokkos::view_alloc(the_data_compute_queue_,
                                                   Kokkos::WithoutInitializing,
                                                   "M"),
                                a_width,
                                a_width}
                , the_vector_b_{Kokkos::view_alloc(the_data_compute_queue_,
                                                   Kokkos::WithoutInitializing,
                                                   "b"),
                                a_width} {
                // EMPTY
            }

            void Multiply() const {
                VectorType<typename PolicyType::FloatType> a_result{Kokkos::view_alloc(the_data_compute_queue_,
                                                                                       Kokkos::WithoutInitializing,
                                                                                       "b"),
                                                                    the_vector_b_.extent(0)};
                // All the views are initialized
                the_data_compute_queue_.fence();

                DoMultiply(the_data_compute_queue_,
                           the_matrix_A_,
                           the_vector_b_,
                           a_result);

                std::cout << "fence0" << std::endl;
                Kokkos::fence();
                std::cout << "fence1" << std::endl;
                the_data_compute_queue_.fence();

                // create_mirror_view will not allocate storage on the host, if
                // the device view (a_result) can be access from the host. Else
                // It has the same behavior has create_mirror.
                auto a_host_result = Kokkos::create_mirror(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{},
                                                                              Kokkos::WithoutInitializing
                                                                              // , "host_result"
                                                                              ),
                                                           a_result);

                Kokkos::deep_copy(the_data_compute_queue_, a_host_result, a_result);

                std::cout << "fence1" << std::endl;
                the_data_compute_queue_.fence();

                for(int i = 0; i < 10; ++i) {
                    std::cout << a_host_result(i) << "; ";
                }
                std::cout << "\n";
            }

            template <typename Queue,
                      typename Matrix,
                      typename Vector0,
                      typename Vector1>
            static void DoMultiply(const Queue&   a_queue,
                                   const Matrix&  the_matrix_A,
                                   const Vector0& the_vector_b,
                                   Vector1&       a_result) {

                auto policy_with_hint = Kokkos::Experimental::require(Kokkos::RangePolicy<Queue>(a_queue,
                                                                                                 0,
                                                                                                 1024 * 1024 * 1024),
                                                                      Kokkos::Experimental::WorkItemProperty::None);

                const auto Compute = KOKKOS_LAMBDA(int n) {
                    const auto extent = 1024 * 64;

                    // using Float          = float2;
                    // Float       value_0  = Float{a_result((n + 0) % extent), a_result((n + 1) % extent)};
                    // Float       value_1  = Float{a_result((n + 2) % extent), a_result((n + 3) % extent)};
                    // Float       value_2  = Float{a_result((n + 4) % extent), a_result((n + 5) % extent)};
                    // Float       value_3  = Float{a_result((n + 6) % extent), a_result((n + 7) % extent)};
                    // Float       value_4  = Float{a_result((n + 8) % extent), a_result((n + 9) % extent)};
                    // Float       value_5  = Float{a_result((n + 10) % extent), a_result((n + 11) % extent)};
                    // Float       value_6  = Float{a_result((n + 12) % extent), a_result((n + 13) % extent)};
                    // Float       value_7  = Float{a_result((n + 14) % extent), a_result((n + 15) % extent)};
                    // const Float constant = Float{1.0F, 1.0F} * a_result((n + 16) % extent);

                    using Float          = float; // or double
                    Float       value_0  = Float(a_result((n + 0) % extent));
                    Float       value_1  = Float(a_result((n + 1) % extent));
                    Float       value_2  = Float(a_result((n + 2) % extent));
                    Float       value_3  = Float(a_result((n + 3) % extent));
                    Float       value_4  = Float(a_result((n + 4) % extent));
                    Float       value_5  = Float(a_result((n + 5) % extent));
                    Float       value_6  = Float(a_result((n + 6) % extent));
                    Float       value_7  = Float(a_result((n + 7) % extent));
                    const Float constant = a_result((n + 8) % extent);

                    // double: 512
                    // float:  512
                    // float2: 1024
                    for(std::size_t i = 0; i < 512; ++i) {
                        value_0 += value_0 * constant;
                        value_1 += value_1 * constant;
                        value_2 += value_2 * constant;
                        value_3 += value_3 * constant;
                        value_4 += value_4 * constant;
                        value_5 += value_5 * constant;
                        value_6 += value_6 * constant;
                        value_7 += value_7 * constant;
                    }

                    value_0 = ((value_0 + value_1) + (value_2 + value_3)) +
                              ((value_4 + value_5) + (value_6 + value_7));

                    // a_result((n + 0) % extent) = value_0.data[0];
                    // a_result((n + 1) % extent) = value_0.data[1];

                    a_result((n + 0) % extent) = value_0;
                };

                Kokkos::parallel_for("mv",
                                     policy_with_hint,
                                     Compute);
            }

        protected:
            Kokkos::DefaultExecutionSpace              the_data_compute_queue_;
            MatrixType<typename PolicyType::FloatType> the_matrix_A_;
            VectorType<typename PolicyType::FloatType> the_vector_b_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif
