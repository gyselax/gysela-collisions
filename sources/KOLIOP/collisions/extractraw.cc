// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <string>

#include "KOLIOP/collisions.h"
#include "KOLIOP/collisions/constants.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/copy.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collisions {

        void Operator::ExtractRaw(Diagnostics value_identifier,
                                  void*       an_output_buffer) const {

            switch(value_identifier) {
                case Diagnostics::k_nustar_coeff: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                nustar_coeff_,
                                                koliop::utility::View::UDL3<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(r_extent_)});
                    break;
                }
                case Diagnostics::k_n_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                n_s_,
                                                koliop::utility::View::UDL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_),
                                                                                   static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case Diagnostics::k_U_par_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                U_par_s_,
                                                koliop::utility::View::UDL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_),
                                                                                   static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case Diagnostics::k_T_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex], T_s_,
                                                koliop::utility::View::UDL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_),
                                                                                   static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case Diagnostics::k_nu_ab: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                nu_ab_,
                                                koliop::utility::View::UDL5<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_),
                                                                                   static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_)});
                    break;
                }
                case Diagnostics::k_q_par_on_nT_s: {
                    koliop::utility::Copy::Deep(execution_space_pool_[Constant::kDefaultExecutionSpacePoolIndex],
                                                q_par_on_nT_s_,
                                                koliop::utility::View::UDL4<Flo64>{static_cast<Flo64*>(an_output_buffer),
                                                                                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_),
                                                                                   static_cast<std::size_t>(species_extent_)});
                    break;
                }

                default:
                    throw std::domain_error{"ExtractRaw: Invalid value_identifier:" + std::to_string(static_cast<int>(value_identifier))};
                    break;
            }
        }

    } // namespace collisions
} // namespace koliop
