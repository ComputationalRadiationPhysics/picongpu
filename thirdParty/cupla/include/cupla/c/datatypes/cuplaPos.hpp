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

struct cuplaPos{
    size_t x, y, z;

    cuplaPos() = default;

    ALPAKA_FN_HOST_ACC
    cuplaPos(
        size_t const x_in,
        size_t const y_in,
        size_t const z_in
    ) :
        x( x_in ),
        y( y_in ),
        z( z_in )
    {}

    template<
      typename TDim,
      typename TSize,
      typename = typename std::enable_if<
          (TDim::value == 3u)
      >::type
    >
    ALPAKA_FN_HOST_ACC
    cuplaPos(
        ::alpaka::Vec<
            TDim,
            TSize
        > const &vec
    )
    {
        for( uint32_t i(0); i < 3u; ++i ) {
            // alpaka vectors are z,y,x.
            ( &this->x )[ i ] = vec[ ( 3u - 1u ) - i ];
        }
    }

    ALPAKA_FN_HOST_ACC
    operator ::alpaka::Vec<
        cupla::AlpakaDim< 3u >,
        cupla::MemSizeType
    >(void) const
    {
        ::alpaka::Vec<
            cupla::AlpakaDim< 3u >,
            cupla::MemSizeType
        > vec( x, y, z );
        return vec;
    }
};

} //namespace CUPLA_ACCELERATOR_NAMESPACE

namespace alpaka
{
namespace traits
{

    //! dimension get trait specialization
    template<>
    struct DimType<
        cuplaPos
    >{
      using type = ::alpaka::DimInt<3u>;
    };

} // namespace traits

namespace traits
{

    //! element type trait specialization
    template<>
    struct ElemType<
        cuplaPos
    >{
        using type = cupla::MemSizeType;
    };

} // namespace traits

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
        cuplaPos,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{

        ALPAKA_FN_HOST_ACC
        static auto
        getExtent( cuplaPos const & extents )
        -> cupla::MemSizeType {
        return (&extents.x)[(3u - 1u) - T_Idx::value];
      }
    };

    //! extent set trait specialization
    template<
        typename T_Idx,
        typename T_Pos
    >
    struct SetExtent<
        T_Idx,
        cuplaPos,
        T_Pos,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setExtent(
            cuplaPos &extents,
            T_Pos const &extent
        )
        -> void
        {
            (&extents.x)[(3u - 1u) - T_Idx::value] = extent;
        }
    };
} // namespace traits
} // namespace extent

namespace traits
{

    //! offset get trait specialization
    template<
        typename T_Idx
    >
    struct GetOffset<
        T_Idx,
        cuplaPos,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        getOffset( cuplaPos const & offsets )
        -> cupla::MemSizeType{
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
        cuplaPos,
        T_Offset,
        typename std::enable_if<
            (3u > T_Idx::value)
        >::type
    >{
        ALPAKA_FN_HOST_ACC
        static auto
        setOffset(
            cuplaPos &offsets,
            T_Offset const &offset
        )
        -> void {
            offsets[(3u - 1u) - T_Idx::value] = offset;
        }
    };
} // namespace traits

namespace traits
{

    //! size type trait specialization.
    template<>
    struct IdxType<
        cuplaPos
    >{
        using type = cupla::MemSizeType;
    };

} // namespace traits
} // namespave alpaka
