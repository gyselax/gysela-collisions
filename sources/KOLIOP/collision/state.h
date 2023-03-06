// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_STATE_H
#define KOLIOP_COLLISION_STATE_H

#include <Kokkos_Core.hpp>


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
            State()
                : the_matrix_A_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "a_thing_"),
                                1024, 1024}
                , the_vector_b_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), 1024} {
                // EMPTY
            }

        protected:
            MatrixType<typename PolicyType::FloatType> the_matrix_A_;
            VectorType<typename PolicyType::FloatType> the_vector_b_;
        };


        ////////////////////////////////////////////////////////////////////////
        // State methods definition
        ////////////////////////////////////////////////////////////////////////

    } // namespace collision
} // namespace koliop

#endif

// Execution Spaces:
//     Where you physically execute
//     Memory Spaces:
//         Where you physically store
//         Memory Layout:
//             How to index
//         Memory Traits:
//             Modifier: stream, random, atomic, texture memory
// Execution Patterns:
//     Parallel for/scan/reduce
// Execution Policies:
//     Range, team, dynamic or static scheduling


