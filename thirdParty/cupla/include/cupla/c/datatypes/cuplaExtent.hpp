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
    struct cuplaExtent
    {
        cupla::MemSizeType width, height, depth;

        cuplaExtent() = default;

        ALPAKA_FN_HOST_ACC
        cuplaExtent(cupla::MemSizeType const w, cupla::MemSizeType const h, cupla::MemSizeType const d)
            : width(w)
            , height(h)
            , depth(d)
        {
        }

        template<typename TDim, typename TSize, typename = typename std::enable_if<(TDim::value == 3u)>::type>
        ALPAKA_FN_HOST_ACC cuplaExtent(::alpaka::Vec<TDim, TSize> const& vec)
        {
            for(uint32_t i(0); i < 3u; ++i)
            {
                // alpaka vectors are z,y,x.
                (&this->width)[i] = vec[(3u - 1u) - i];
            }
        }

        ALPAKA_FN_HOST_ACC
        operator ::alpaka::Vec<cupla::AlpakaDim<3u>, cupla::MemSizeType>(void) const
        {
            ::alpaka::Vec<cupla::AlpakaDim<3u>, cupla::MemSizeType> vec(depth, height, width);
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
        struct DimType<cuplaExtent>
        {
            using type = ::alpaka::DimInt<3u>;
        };

        //! element type trait specialization
        template<>
        struct ElemType<cuplaExtent>
        {
            using type = cupla::MemSizeType;
        };

        //! offset get trait specialization
        template<>
        struct GetOffsets<cuplaExtent>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(cuplaExtent const& offsets) -> Vec<::alpaka::DimInt<3u>, cupla::MemSizeType>
            {
                return {offsets.depth, offsets.height, offsets.width};
            }
        };

        //! size type trait specialization.
        template<>
        struct IdxType<cuplaExtent>
        {
            using type = cupla::MemSizeType;
        };

    } // namespace trait
} // namespace alpaka
