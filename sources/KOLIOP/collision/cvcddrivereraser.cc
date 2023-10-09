// SPDX-License-Identifier: MIT

#include "KOLIOP/collision/cvcddrivereraser.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverEraser methods definition
        ////////////////////////////////////////////////////////////////////////

        CvCdDriverEraser::InterfaceType*
        CvCdDriverEraser::get() {
            return the_erased_cvcd_driver_.get();
        }

        CvCdDriverEraser::InterfaceType&
        CvCdDriverEraser::operator*() {
            return *the_erased_cvcd_driver_.get();
        }

        CvCdDriverEraser::InterfaceType*
        CvCdDriverEraser::operator->() {
            return the_erased_cvcd_driver_.get();
        }

        void CvCdDriverEraser::clear() {
            the_erased_cvcd_driver_.reset(nullptr);
        }

        bool CvCdDriverEraser::IsSet() const {
            return the_erased_cvcd_driver_.get() != nullptr;
        }

    } // namespace collision
} // namespace koliop
