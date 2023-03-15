// SPDX-License-Identifier: MIT

#ifndef KOLIOP_PORTABILITY_CMATH_H
#define KOLIOP_PORTABILITY_CMATH_H

#include <cmath>

#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace portability {

        template <typename ArithmeticEnabled>
        struct GyselaConstant {
        public:
            using value_type = ArithmeticEnabled;

            static_assert(std::is_arithmetic_v<value_type>, "");

        public:
            static inline constexpr value_type kInfinity = static_cast<value_type>(1.0E+30);
        };

        template <typename ArithmeticEnabled>
        struct MathematicalConstant {
        public:
            using value_type = ArithmeticEnabled;

            static_assert(std::is_arithmetic_v<value_type>, "");

        public:
            static inline constexpr value_type kE        = static_cast<value_type>(2.71828182845904523536);  // e
            static inline constexpr value_type kLog2E    = static_cast<value_type>(1.44269504088896340736);  // log2(e)
            static inline constexpr value_type kLog10E   = static_cast<value_type>(0.434294481903251827651); // log10(e)
            static inline constexpr value_type kLn2      = static_cast<value_type>(0.693147180559945309417); // ln(2)
            static inline constexpr value_type kLn10     = static_cast<value_type>(2.30258509299404568402);  // ln(10)
            static inline constexpr value_type kPi       = static_cast<value_type>(3.14159265358979323846);  // pi
            static inline constexpr value_type kPi_2     = static_cast<value_type>(1.57079632679489661923);  // pi/2
            static inline constexpr value_type kPi_4     = static_cast<value_type>(0.785398163397448309616); // pi/4
            static inline constexpr value_type k1_Pi     = static_cast<value_type>(0.318309886183790671538); // 1/pi
            static inline constexpr value_type k2_Pi     = static_cast<value_type>(0.636619772367581343076); // 2/pi
            static inline constexpr value_type k2_SQRTPI = static_cast<value_type>(1.12837916709551257390);  // 2/sqrt(pi)
            static inline constexpr value_type kSQRT2    = static_cast<value_type>(1.41421356237309504880);  // sqrt(2)
            static inline constexpr value_type k1_SQRT2  = static_cast<value_type>(0.707106781186547524401); // 1/sqrt(2) or sqrt(1/2)
        };

    } // namespace portability
} // namespace koliop

#endif
