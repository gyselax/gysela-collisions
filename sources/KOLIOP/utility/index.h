// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_INDEX_H
#define KOLIOP_UTILITY_INDEX_H

namespace koliop {
    namespace utility {
        /// @brief Inspired from libsplb2:utility/isomap
        namespace mapper {
            /// NOTE: To fix dimension while keeping some dynamic proceed like so:
            /// global=Cartesian{3,5,7,11};
            /// dynmic=global.Slice(1,2);
            /// static=global.Slice(0,3);
            /// Typical accessing:
            /// for(i1,i2)
            ///     data[global(i0,i1,i2,i3)] -> recompute each time or be at the mercy of the compiler.
            /// With Strided mappers:
            /// static_offset=static(i0, i3)
            /// for(i1,i2)
            ///     dynmic_offset=dynmic(i1, i2)
            ///     data[static_offset+dynmic_offset]
            template <typename OrdinalType,
                      int kDimensions>
            struct Strided {
            protected:
                using Holder = std::array<OrdinalType, kDimensions>;

            public:
                template <typename... OrdinalArgs>
                constexpr Strided(OrdinalArgs... strides)
                    : strides_{static_cast<OrdinalType>(strides)...} {
                    static_assert(sizeof...(strides) == kDimensions);
                }
                template <typename... OrdinalArgs>
                constexpr OrdinalType operator()(OrdinalArgs... coordinates) const {
                    static_assert(sizeof...(coordinates) == kDimensions);
                    const Holder coordinates_array{static_cast<OrdinalType>(coordinates)...};
                    OrdinalType  offset = 0;
                    for(int idim = 0; idim < kDimensions; ++idim) {
                        offset += coordinates_array[idim] * strides_[idim];
                    }
                    return offset;
                }

            protected:
                Holder strides_;
            };

            /// Offset computation with strides build from left to right prefix.
            ///
            template <typename OrdinalType,
                      int kDimensions>
            class Cartesian {
            protected:
                using Holder = std::array<OrdinalType, kDimensions>;

            public:
                template <typename... OrdinalArgs>
                constexpr Cartesian(OrdinalArgs... extents)
                    : extents_{static_cast<OrdinalType>(extents)...} {
                    static_assert(sizeof...(extents) == kDimensions);
                }
                template <typename... OrdinalArgs>
                constexpr OrdinalType operator()(OrdinalArgs... coordinates) const {
                    static_assert(sizeof...(coordinates) == kDimensions);
                    const Holder coordinates_array{static_cast<OrdinalType>(coordinates)...};
                    OrdinalType  offset  = 0;
                    const auto   strides = Strides();
                    for(int idim = 0; idim < kDimensions; ++idim) {
                        offset += coordinates_array[idim] * strides[idim];
                    }
                    return offset;
                }
                constexpr auto Inverse(OrdinalType flat_index) const {
                    Holder     coordinates{};
                    const auto strides = Strides();

                    int idim = 0;
                    for(; idim < (kDimensions - 1); ++idim) {
                        coordinates[idim] = (flat_index / strides[idim]) % Extent(idim);
                    }
                    coordinates[idim] = (flat_index / strides[idim]) /*% Extent(idim)*/;
                    return coordinates;
                }
                template <typename... IndiciesArgs>
                constexpr auto Swizzle(IndiciesArgs... dimension_indicies) const {
                    return Cartesian<OrdinalType, sizeof...(dimension_indicies)>{
                        Extent(dimension_indicies)...};
                }
                constexpr OrdinalType Extent(int idim) const {
                    return extents_[idim];
                }
                constexpr Holder Strides() const {
                    Holder strides{};
                    strides[0] = 1;
                    for(int idim = 1; idim < kDimensions; ++idim) {
                        strides[idim] = strides[idim - 1] * Extent(idim - 1);
                    }
                    return strides;
                }
                constexpr OrdinalType Cardinality() const {
                    OrdinalType cardinality = 1;
                    for(int idim = 0; idim < kDimensions; ++idim) {
                        cardinality *= Extent(idim);
                    }
                    return cardinality;
                }
                template <typename... IndiciesArgs>
                constexpr auto Slice(IndiciesArgs... dimension_indicies) const {
                    const auto strides = Strides();
                    return Strided<OrdinalType, sizeof...(dimension_indicies)>{
                        strides[dimension_indicies]...};
                }

            protected:
                Holder extents_;
            };
        } // namespace mapper
    } // namespace utility
} // namespace koliop

#endif
