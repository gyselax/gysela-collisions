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

        /// @brief Hides the compiletime blocking of the CvCd operation. This
        /// allows us to have compiletime blocking and a runtime dispatch on
        /// these blocks.
        ///
        class CvCdDriverEraser {
        public:
            class InterfaceType {
            public:
            public:
                virtual void AllocateScratchMemory() = 0;
                virtual void ReleaseScratchMemory()  = 0;

                virtual void Apply(const Operator&                          an_operator,
                                   Constant::DeviceExecutionSpaceType       an_execution_space,
                                   OrdinalType                              block_first_tor1,
                                   OrdinalType                              block_first_tor2,
                                   OrdinalType                              block_first_tor3,
                                   ComputeFloatType                         dt,
                                   OrdinalType                              species_index,
                                   koliop::utility::View::MDL<Flo64******>& f5d_s)       = 0;
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
                    : value_{std::move(a_value)} {
                    // EMPTY
                }

                void AllocateScratchMemory() override {
                    value_.AllocateScratchMemory();
                }

                void ReleaseScratchMemory() override {
                    value_.ReleaseScratchMemory();
                }

                void Apply(const Operator&                          an_operator,
                           Constant::DeviceExecutionSpaceType       an_execution_space,
                           OrdinalType                              block_first_tor1,
                           OrdinalType                              block_first_tor2,
                           OrdinalType                              block_first_tor3,
                           ComputeFloatType                         dt,
                           OrdinalType                              species_index,
                           koliop::utility::View::MDL<Flo64******>& f5d_s) override {
                    value_.Apply(an_operator,
                                 an_execution_space,
                                 block_first_tor1,
                                 block_first_tor2,
                                 block_first_tor3,
                                 dt,
                                 species_index,
                                 f5d_s);
                }

            protected:
                value_type value_;
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
            std::shared_ptr<InterfaceType> erased_cvcd_driver_;
        };


        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverEraser methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename T>
        CvCdDriverEraser&
        CvCdDriverEraser::Set(T&& a_cvcd_driver) {
            erased_cvcd_driver_ = std::make_shared<ImplementationType<
                std::remove_reference_t<
                    std::remove_cv_t<T>>>>(std::forward<T>(a_cvcd_driver));
            return *this;
        }

    } // namespace collision
} // namespace koliop

#endif
