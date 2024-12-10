// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <string>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/collision/operator.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/koliop.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        void Operator::ExtractRaw(uint32_t value_identifier,
                                  void*    an_output_buffer) const {

            switch(value_identifier) {
                case KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                nustar_coeff_,
                                                koliop::utility::View::ExternalInputL3<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                              static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(r_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_n_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                n_s_,
                                                koliop::utility::View::ExternalInputL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                              static_cast<std::size_t>(r_extent_), static_cast<std::size_t>(theta_extent_), static_cast<std::size_t>(phi_extent_),
                                                                                              static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_U_par_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                U_par_s_,
                                                koliop::utility::View::ExternalInputL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                              static_cast<std::size_t>(r_extent_), static_cast<std::size_t>(theta_extent_), static_cast<std::size_t>(phi_extent_),
                                                                                              static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_T_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex], T_s_,
                                                koliop::utility::View::ExternalInputL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                              static_cast<std::size_t>(r_extent_), static_cast<std::size_t>(theta_extent_), static_cast<std::size_t>(phi_extent_),
                                                                                              static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_nu_ab: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                nu_ab_,
                                                koliop::utility::View::ExternalInputL5<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                              static_cast<std::size_t>(r_extent_), static_cast<std::size_t>(theta_extent_), static_cast<std::size_t>(phi_extent_),
                                                                                              static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                q_par_on_nT_s_,
                                                koliop::utility::View::ExternalInputL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                              static_cast<std::size_t>(r_extent_), static_cast<std::size_t>(theta_extent_), static_cast<std::size_t>(phi_extent_),
                                                                                              static_cast<std::size_t>(species_extent_)});
                    break;
                }

                default:
                    throw std::domain_error{"ExtractRaw: Invalid value_identifier:" + std::to_string(value_identifier)};
                    break;
            }

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
