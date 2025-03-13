// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_MATH_H
#define KOLIOP_UTILITY_MATH_H

#include <cstdint>

namespace koliop {
    namespace utility {

        constexpr std::uint32_t CheapLog2(std::uint64_t value) {
            // This function compiles to a bsr/xor/add

            std::uint32_t count = 0;

            value >>= 1;

            while(value != 0) {
                value >>= 1;
                ++count;
            }

            return count;
        }

    } // namespace utility
} // namespace koliop

#endif
