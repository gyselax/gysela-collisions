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
                                                                                       Kokkos::WithoutInitializing, "b"),
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
                Kokkos::parallel_for(
                    "mv",
                    Kokkos::RangePolicy<Queue>(a_queue,
                                               0,
                                               the_matrix_A.extent(0)),
                    KOKKOS_LAMBDA(int n) {
                        auto sum = the_vector_b(0) * the_matrix_A(n, 0);
                        for(std::size_t i = 1; i < the_matrix_A.extent(1); ++i) {
                            sum += the_vector_b(i) * the_matrix_A(n, i);
                        }
                        a_result(n) = sum;
                    });
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
