// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "KOLIOP/collision/operator.h"
#include "KOLIOP/koliop.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collision {

        void Operator::ExtractRaw(uint32_t the_value_identifier,
                                  void*    an_output_buffer) const {

            switch(the_value_identifier) {
                case KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff: {
                    koliop::utility::Copy::Deep(nustar_coeff_,
                                                koliop::utility::View::UHL3<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_r_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_n_s: {
                    koliop::utility::Copy::Deep(n_s_,
                                                koliop::utility::View::UHL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                                   static_cast<std::size_t>(the_species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_U_par_s: {
                    koliop::utility::Copy::Deep(U_par_s_,
                                                koliop::utility::View::UHL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                                   static_cast<std::size_t>(the_species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_T_s: {
                    koliop::utility::Copy::Deep(T_s_,
                                                koliop::utility::View::UHL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                                   static_cast<std::size_t>(the_species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_nu_ab: {
                    koliop::utility::Copy::Deep(nu_ab_,
                                                koliop::utility::View::UHL5<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                                   static_cast<std::size_t>(the_species_extent_), static_cast<std::size_t>(the_species_extent_)});
                    break;
                }
                case KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s: {
                    koliop::utility::Copy::Deep(q_par_on_nT_s_,
                                                koliop::utility::View::UHL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(the_r_extent_), static_cast<std::size_t>(the_theta_extent_), static_cast<std::size_t>(the_phi_extent_),
                                                                                   static_cast<std::size_t>(the_species_extent_)});
                    break;
                }

                default:
                    throw std::domain_error{"ExtractRaw: Invalid the_value_identifier:" + std::to_string(the_value_identifier)};
                    break;
            }

#if defined(KOLIOP_FENCE_ON_OPERATOR_EXIT)
            Kokkos::fence();
#endif
        }

    } // namespace collision
} // namespace koliop
