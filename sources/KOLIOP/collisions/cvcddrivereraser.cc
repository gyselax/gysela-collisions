// SPDX-License-Identifier: MIT

#include "KOLIOP/collisions/cvcddrivereraser.h"

namespace koliop {
    namespace collisions {

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverEraser methods definition
        ////////////////////////////////////////////////////////////////////////

        CvCdDriverEraser::InterfaceType*
        CvCdDriverEraser::get() {
            return erased_cvcd_driver_.get();
        }

        const CvCdDriverEraser::InterfaceType*
        CvCdDriverEraser::get() const {
            return erased_cvcd_driver_.get();
        }

        CvCdDriverEraser::InterfaceType&
        CvCdDriverEraser::operator*() {
            return *get();
        }

        CvCdDriverEraser::InterfaceType*
        CvCdDriverEraser::operator->() {
            return get();
        }

        void CvCdDriverEraser::clear() {
            // NOTE: there is an asymmetry where a unique_ptr's reset can take a
            // nullptr but not a shared_ptr's.
            erased_cvcd_driver_.reset(/* nullptr */);
        }

        bool CvCdDriverEraser::IsSet() const {
            return get() != nullptr;
        }

    } // namespace collisions
} // namespace koliop
