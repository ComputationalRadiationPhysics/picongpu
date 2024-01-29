/* Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
    struct cuplaPos
    {
        size_t x, y, z;

        cuplaPos() = default;

        ALPAKA_FN_HOST_ACC
        cuplaPos(size_t const x_in, size_t const y_in, size_t const z_in) : x(x_in), y(y_in), z(z_in)
        {
        }

        template<typename TDim, typename TSize, typename = typename std::enable_if<(TDim::value == 3u)>::type>
        ALPAKA_FN_HOST_ACC cuplaPos(::alpaka::Vec<TDim, TSize> const& vec)
        {
            for(uint32_t i(0); i < 3u; ++i)
            {
                // alpaka vectors are z,y,x.
                (&this->x)[i] = vec[(3u - 1u) - i];
            }
        }

        ALPAKA_FN_HOST_ACC
        operator ::alpaka::Vec<cupla::AlpakaDim<3u>, cupla::MemSizeType>(void) const
        {
            ::alpaka::Vec<cupla::AlpakaDim<3u>, cupla::MemSizeType> vec(x, y, z);
            return vec;
        }
    };

} // namespace CUPLA_ACCELERATOR_NAMESPACE

namespace alpaka
{
    namespace trait
    {
        //! dimension get trait specialization
        template<>
        struct DimType<cuplaPos>
        {
            using type = ::alpaka::DimInt<3u>;
        };

        //! element type trait specialization
        template<>
        struct ElemType<cuplaPos>
        {
            using type = cupla::MemSizeType;
        };

        //! extent get trait specialization
        template<>
        struct GetExtents<cuplaPos>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(cuplaPos const& extents) -> Vec<::alpaka::DimInt<3u>, cupla::MemSizeType>
            {
                return {extents.z, extents.y, extents.x};
            }
        };

        //! offset get trait specialization
        template<>
        struct GetOffsets<cuplaPos>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(cuplaPos const& offsets) -> Vec<::alpaka::DimInt<3u>, cupla::MemSizeType>
            {
                return {offsets.z, offsets.y, offsets.x};
            }
        };

        //! size type trait specialization.
        template<>
        struct IdxType<cuplaPos>
        {
            using type = cupla::MemSizeType;
        };

    } // namespace trait
} // namespace alpaka
