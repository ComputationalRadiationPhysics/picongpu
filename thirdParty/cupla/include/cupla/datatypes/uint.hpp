/**
 * Copyright 2015-2016 Rene Widera
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

#include "cupla/types.hpp"

namespace cupla
{

    struct uint3{
        IdxType x, y, z;

        ALPAKA_FN_HOST_ACC
        uint3() = default;

        template<
          typename TDim,
          typename TSize,
          typename = typename std::enable_if<
              (TDim::value == 3u)
          >::type
        >
        ALPAKA_FN_HOST_ACC
        uint3(
          ::alpaka::vec::Vec<
              TDim,
              TSize
          > const &vec
        ){
            for (uint32_t i(0); i < 3u; ++i) {
                // alpaka vectors are z,y,x.
                (&(this->x))[i] = vec[(3u - 1u) - i];
            }
        }

#if( ALPAKA_ACC_GPU_CUDA_ENABLED == 1 )
        ALPAKA_FN_HOST_ACC
        uint3(
          ::uint3 const & vec
        ){
            for (uint32_t i(0); i < 3u; ++i) {
                (&(this->x))[i] = (&(vec.x))[i];
            }
        }
#endif

        ALPAKA_FN_HOST_ACC
        operator ::alpaka::vec::Vec<
            cupla::AlpakaDim< 3u >,
            IdxType
        >(void) const
        {
            ::alpaka::vec::Vec<
                cupla::AlpakaDim< 3u >,
                IdxType
            > vec(z, y, x);
            return vec;
        }
    };

} //namespace cupla


namespace alpaka
{
namespace dim
{
namespace traits
{

    //! dimension get trait specialization
    template<>
    struct DimType<
        cupla::uint3
    >{
      using type = ::alpaka::dim::DimInt<3u>;
    };

} // namespace traits
} // namespace dim

namespace elem
{
namespace traits
{

    //! element type trait specialization
    template<>
    struct ElemType<
        cupla::uint3
    >{
        using type = cupla::IdxType;
    };

} // namespace traits
} // namspace elem

namespace extent
{
namespace traits
{

    //! extent get trait specialization
    template<
        typename T_Idx
    >
    struct GetExtent<
        T_Idx,
        cupla::uint3,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{

        ALPAKA_FN_HOST_ACC
        static auto
        getExtent( cupla::uint3 const &extents )
        -> cupla::IdxType {
        return (&extents.x)[(3u - 1u) - T_Idx::value];
      }
    };

    //! extent set trait specialization
    template<
        typename T_Idx,
        typename T_Extent
    >
    struct SetExtent<
        T_Idx, cupla::uint3,
        T_Extent,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setExtent(
            cupla::uint3 &extents,
            T_Extent const &extent
        )
        -> void
        {
            (&extents.x)[(3u - 1u) - T_Idx::value] = extent;
        }
    };
} // namespace traits
} // namespace extent

namespace offset
{
namespace traits
{

    //! offset get trait specialization
    template<
        typename T_Idx
    >
    struct GetOffset<
        T_Idx,
        cupla::uint3,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        getOffset( cupla::uint3 const & offsets )
        -> cupla::IdxType{
            return (&offsets.x)[(3u - 1u) - T_Idx::value];
        }
    };


    //! offset set trait specialization.
    template<
        typename T_Idx,
        typename T_Offset
    >
    struct SetOffset<
        T_Idx,
        cupla::uint3,
        T_Offset,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setOffset(
            cupla::uint3 &offsets,
            T_Offset const &offset
        )
        -> void {
            offsets[(3u - 1u) - T_Idx::value] = offset;
        }
    };
} // namespace traits
} // namespace offset

namespace size
{
namespace traits
{

    //! size type trait specialization.
    template<>
    struct SizeType<
        cupla::uint3
    >{
        using type = cupla::IdxType;
    };

} // namespace traits
} // namespace size
} // namespave alpaka
