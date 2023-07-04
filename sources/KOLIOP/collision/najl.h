// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_NAJL_H
#define KOLIOP_COLLISION_NAJL_H

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Najl definition
        ////////////////////////////////////////////////////////////////////////

        /// Computes the Najl coefficients.
        /// The N0 coefficients are stored in a
        /// ScalarBlockTridiagonalByBlockMatrixType. See
        /// BatchedBlockTridiagonalByBlockTRF for the layout. Note that the Najl
        /// coefficients are accessed by selecting the proper diagonal of the
        /// matrix representation.
        ///
        /// For instance:
        /// ScalarMatrixType{}(5, 4, 2, 2, 1, 0)
        /// represents the first (0) batch element l=1 and j=2
        /// of the N2 (2) coefficient block at the fifth vpar
        /// (4) of the sixth batch (5).
        /// And similary:
        /// ScalarMatrixType{}(5, 4, 1, 2, 1, 0) is the same thing but for N1.
        /// ScalarMatrixType{}(5, 4, 0, 2, 1, 0) is the same thing but for N0.
        ///
        class Najl {
        public:
            using ExecutionSpaceType = typename Constant::DeviceExecutionSpaceType;

            /// The Najl coefficients are stored in a data structure similar to
            /// how the one used for the matrix in Operator::Apply().
            ///
            using MatrixKindType = Constant::MatrixKindType;

        public:
            Najl(OrdinalType                                          pSpecies_id,
                 OrdinalType                                          the_vpar_extent,
                 OrdinalType                                          the_species_extent,
                 bool                                                 collision_interspecies,
                 const Flo64*                                         comb_mat,
                 koliop::utility::View::InternalInputL1<const Flo64>& hat_As,
                 koliop::utility::View::InternalInputL1<const Flo64>& hat_Zs,
                 koliop::utility::View::InternalInputL1<const Flo64>& vparg);

            void Compute(ExecutionSpaceType                                   an_execution_space,
                         OrdinalType                                          the_block_first_r,
                         OrdinalType                                          the_block_first_theta,
                         OrdinalType                                          the_block_first_phi,
                         const koliop::utility::View::MDD4<ComputeFloatType>& n_s,
                         const koliop::utility::View::MDD4<ComputeFloatType>& T_s,
                         const koliop::utility::View::MDD5<ComputeFloatType>& nu_ab,
                         typename MatrixKindType::ScalarType&                 the_Najl) const;

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            // Constant, externally computed values

            OrdinalType pSpecies_id_;
            OrdinalType the_vpar_extent_;
            OrdinalType the_species_extent_;
            bool        collision_interspecies_;

            koliop::utility::View::InternalInputL2<const Flo64> comb_mat_; // TODO(Etienne M): constant extent.
            koliop::utility::View::InternalInputL1<const Flo64> hat_As_;
            koliop::utility::View::InternalInputL1<const Flo64> hat_Zs_;
            koliop::utility::View::InternalInputL1<const Flo64> vparg_;

            // Constant, internally computed values

            koliop::utility::View::MDD3<ComputeFloatType> C_jl_; // TODO(Etienne M): constant extent.
        };

    } // namespace collision
} // namespace koliop

#endif
