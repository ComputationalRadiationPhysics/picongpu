/* Copyright 2015-2016 Rene Widera
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

namespace cupla
{
    inline namespace CUPLA_ACCELERATOR_NAMESPACE
    {
        struct uint3
        {
            IdxType x, y, z;

            uint3() = default;

            template<typename TDim, typename TSize, typename = typename std::enable_if<(TDim::value == 3u)>::type>
            ALPAKA_FN_HOST_ACC uint3(::alpaka::Vec<TDim, TSize> const& vec)
            {
                for(uint32_t i(0); i < 3u; ++i)
                {
                    // alpaka vectors are z,y,x.
                    (&(this->x))[i] = vec[(3u - 1u) - i];
                }
            }

#if(ALPAKA_ACC_GPU_CUDA_ENABLED == 1 || ALPAKA_ACC_GPU_HIP_ENABLED == 1)
            ALPAKA_FN_HOST_ACC
            uint3(::uint3 const& vec)
            {
                for(uint32_t i(0); i < 3u; ++i)
                {
                    (&(this->x))[i] = (&(vec.x))[i];
                }
            }
#endif

            ALPAKA_FN_HOST_ACC
            operator ::alpaka::Vec<cupla::AlpakaDim<3u>, IdxType>(void) const
            {
                ::alpaka::Vec<cupla::AlpakaDim<3u>, IdxType> vec(z, y, x);
                return vec;
            }
        };

    } // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla


namespace alpaka
{
    namespace trait
    {
        //! dimension get trait specialization
        template<>
        struct DimType<cupla::uint3>
        {
            using type = ::alpaka::DimInt<3u>;
        };

        //! element type trait specialization
        template<>
        struct ElemType<cupla::uint3>
        {
            using type = cupla::IdxType;
        };

        //! extent get trait specialization
        template<>
        struct GetExtents<cupla::uint3>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(cupla::uint3 const& extents) -> Vec<::alpaka::DimInt<3u>, cupla::IdxType>
            {
                return {extents.z, extents.y, extents.x};
            }
        };

        //! offset get trait specialization
        template<>
        struct GetOffsets<cupla::uint3>
        {
            ALPAKA_FN_HOST_ACC
            constexpr auto operator()(cupla::uint3 const& offsets) -> Vec<::alpaka::DimInt<3u>, cupla::IdxType>
            {
                return {offsets.z, offsets.y, offsets.x};
            }
        };

        //! size type trait specialization.
        template<>
        struct IdxType<cupla::uint3>
        {
            using type = cupla::IdxType;
        };

    } // namespace trait
} // namespace alpaka
