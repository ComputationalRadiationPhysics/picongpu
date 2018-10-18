/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for fixed idx arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory ops.
    /*namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array device type trait specialization.
            template<
                typename TFixedSizeArray>
            struct DevType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The fixed idx array device get trait specialization.
            template<
                typename TFixedSizeArray>
            struct GetDev<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    TFixedSizeArray const & view)
                -> dev::DevCpu
                {
                    // \FIXME: CUDA device?
                    return pltf::getDevByIdx<pltf::PltfCpu>(0u);
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array dimension getter trait specialization.
            template<
                typename TFixedSizeArray>
            struct DimType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = dim::DimInt<std::rank<TFixedSizeArray>::value>;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array memory element type get trait specialization.
            template<
                typename TFixedSizeArray>
            struct ElemType<
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = typename std::remove_all_extent<TFixedSizeArray>::type;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array width get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TFixedSizeArray>
            struct GetExtent<
                TIdxIntegralConst,
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value > TIdxIntegralConst::value)
                    && (std::extent<TFixedSizeArray, TIdxIntegralConst::value>::value > 0u)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static constexpr auto getExtent(
                    TFixedSizeArray const & //extent
                )
                -> idx::Idx<TFixedSizeArray>
                {
                    // C++14 constexpr with void return
                    //alpaka::ignore_unused(extent);
                    return std::extent<TFixedSizeArray, TIdxIntegralConst::value>::value;
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The fixed idx array native pointer get trait specialization.
                template<
                    typename TFixedSizeArray>
                struct GetPtrNative<
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value>::type>
                {
                    using TElem = typename std::remove_all_extent<TFixedSizeArray>::type;

                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        TFixedSizeArray const & view)
                    -> TElem const *
                    {
                        return view;
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        TFixedSizeArray & view)
                    -> TElem *
                    {
                        return view;
                    }
                };

                //#############################################################################
                //! The fixed idx array pitch get trait specialization.
                template<
                    typename TFixedSizeArray>
                struct GetPitchBytes<
                    dim::DimInt<std::rank<TFixedSizeArray>::value - 1u>,
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value
                        && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
                {
                    using TElem = typename std::remove_all_extent<TFixedSizeArray>::type;

                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static constexpr auto getPitchBytes(
                        TFixedSizeArray const &)
                    -> idx::Idx<TFixedSizeArray>
                    {
                        return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array offset get trait specialization.
            template<
                typename TIdx,
                typename TFixedSizeArray>
            struct GetOffset<
                TIdx,
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    TFixedSizeArray const &)
                -> idx::Idx<TFixedSizeArray>
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector idx type trait specialization.
            template<
                typename TFixedSizeArray>
            struct IdxType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = std::size_t;
            };
        }
    }*/
}
