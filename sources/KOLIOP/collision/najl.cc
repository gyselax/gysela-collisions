// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/najl.h"

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        template <typename View3D,
                  typename MirrorView2D>
        static inline void
        DoPrepareCjl(const MirrorView2D& a_host_visible_C_j,
                     View3D&             C_jl) {

            // a_host_visible_C_jl is zero initialized.
            auto a_host_visible_C_jl = Kokkos::create_mirror_view(C_jl);

            const auto CoreLoop = [&](auto j_tmp, auto l_tmp) {
                for(OrdinalType i = 0; i <= (j_tmp + l_tmp); ++i) {
                    // This seems unnecessary ?
                    a_host_visible_C_jl(i, j_tmp, l_tmp) = 0.0;
                }

                for(OrdinalType alpha = 0; alpha <= j_tmp; ++alpha) {
                    for(OrdinalType beta = 0; beta <= l_tmp; ++beta) {
                        a_host_visible_C_jl(alpha + beta, j_tmp, l_tmp) += a_host_visible_C_j(alpha, j_tmp) *
                                                                           a_host_visible_C_j(beta, l_tmp);
                    }
                }
            };

            for(OrdinalType i = 0; i < Operator::kMaximumLaguerrePolynomialDegree; ++i) {
                for(OrdinalType j = 0; j < Operator::kMaximumLaguerrePolynomialDegree; ++j) {
                    CoreLoop(i, j);
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_jl, C_jl);
        }

        template <typename View3D,
                  typename View2D>
        static inline void
        DoPrepareCj_Cjl(View2D& C_j,
                        View3D& C_jl) {

            // TODO(Etienne M): This small matrix is know at compile time !

            // a_host_visible_C_j is zero initialized.
            auto a_host_visible_C_j = Kokkos::create_mirror_view(C_j);

            for(OrdinalType j = 0; j < Operator::kMaximumLaguerrePolynomialDegree; ++j) {
                a_host_visible_C_j(0, j) = 1.0;
                a_host_visible_C_j(1, j) = static_cast<ComputeFloatType>(-static_cast<int64_t>(j));
            }

            if(Operator::kMaximumLaguerrePolynomialDegree > 2) {
                a_host_visible_C_j(2, 2) = 0.5;
                if(Operator::kMaximumLaguerrePolynomialDegree > 3) {
                    // NOTE: Assuming kMaximumLaguerrePolynomialDegree = 3, this
                    // is never evaluated.
                    a_host_visible_C_j(2, 3) = 1.5;
                    a_host_visible_C_j(3, 3) = -1.0 / 6.0;
                }
            }

            koliop::utility::Copy::Shallow(a_host_visible_C_j, C_j);

            DoPrepareCjl(a_host_visible_C_j, C_jl);
        }

        Najl::Najl()
            : C_j_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_j_"),
                   static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree),
                   static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree)}
            , C_jl_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "C_jl_"),
                    static_cast<std::size_t>(2 * Operator::kMaximumLaguerrePolynomialDegree - 1),
                    static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree),
                    static_cast<std::size_t>(Operator::kMaximumLaguerrePolynomialDegree)} {

            // TODO(Etienne M): Observe how the C_j_ and C_jl_ values are
            // constant. they could be compile time computed or hardcoded.
            DoPrepareCj_Cjl(C_j_, C_jl_);
        }

    } // namespace collision
} // namespace koliop
