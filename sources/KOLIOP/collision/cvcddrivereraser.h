// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_CVCDDRIVERERASER_H
#define KOLIOP_COLLISION_CVCDDRIVERERASER_H

#include <memory>

#include "KOLIOP/collision/constant.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        // NOTE: Ugly declaration..
        class Operator;

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverEraser definition
        ////////////////////////////////////////////////////////////////////////

        class CvCdDriverEraser {
        public:
            class InterfaceType {
            public:
            public:
                virtual void Apply(const Operator&                                an_operator,
                                   Constant::DeviceExecutionSpaceType             an_execution_space,
                                   OrdinalType                                    the_block_first_r,
                                   OrdinalType                                    the_block_first_theta,
                                   OrdinalType                                    the_block_first_phi,
                                   ComputeFloatType                               dt,
                                   OrdinalType                                    the_species_index,
                                   koliop::utility::View::InternalInputL6<Flo64>& f5d_s) = 0;
                virtual ~InterfaceType()                                                 = default;

            protected:
            };

        protected:
            template <typename CvCdDriverType>
            class ImplementationType final : public InterfaceType {
            public:
                using value_type = CvCdDriverType;

            public:
                explicit ImplementationType(value_type&& a_value)
                    : the_value_{std::move(a_value)} {
                    // EMPTY
                }

                void Apply(const Operator&                                an_operator,
                           Constant::DeviceExecutionSpaceType             an_execution_space,
                           OrdinalType                                    the_block_first_r,
                           OrdinalType                                    the_block_first_theta,
                           OrdinalType                                    the_block_first_phi,
                           ComputeFloatType                               dt,
                           OrdinalType                                    the_species_index,
                           koliop::utility::View::InternalInputL6<Flo64>& f5d_s) override {
                    the_value_.Apply(an_operator,
                                     an_execution_space,
                                     the_block_first_r,
                                     the_block_first_theta,
                                     the_block_first_phi,
                                     dt,
                                     the_species_index,
                                     f5d_s);
                }

            protected:
                value_type the_value_;
            };

        public:
            template <typename T>
            CvCdDriverEraser& Set(T&& a_cvcd_driver);

            InterfaceType*       get();
            const InterfaceType* get() const;
            InterfaceType&       operator*();
            InterfaceType*       operator->();

            void clear();

            bool IsSet() const;

        protected:
            std::shared_ptr<InterfaceType> the_erased_cvcd_driver_;
        };


        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverEraser methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename T>
        CvCdDriverEraser&
        CvCdDriverEraser::Set(T&& a_cvcd_driver) {
            the_erased_cvcd_driver_ = std::make_shared<ImplementationType<
                std::remove_reference_t<
                    std::remove_cv_t<T>>>>(std::forward<T>(a_cvcd_driver));
            return *this;
        }

    } // namespace collision
} // namespace koliop

#endif
