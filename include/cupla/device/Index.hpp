/* Copyright 2020 Rene Widera
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

#include "cupla/datatypes/uint.hpp"
#include "cupla/types.hpp"

#include <alpaka/alpaka.hpp>

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
inline namespace device
{

    /** number of blocks within the grid layer
     *
     * @tparam T_Acc alpaka accelerator [alpaka::*]
     * @param acc alpaka accelerator
     */
    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    cupla::uint3 gridDim( T_Acc const & acc )
    {
        return static_cast< uint3 >(
            ::alpaka::getWorkDiv<
                ::alpaka::Grid,
                ::alpaka::Blocks
            >( acc )
        );
    }

    /** number of threads within the block layer
     *
     * @tparam T_Acc alpaka accelerator [alpaka::*]
     * @param acc alpaka accelerator
     */
    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    cupla::uint3 blockDim( T_Acc const & acc )
    {
        return static_cast< uint3 >(
            ::alpaka::getWorkDiv<
                ::alpaka::Block,
                ::alpaka::Threads
            >( acc )
        );
    }

    /** number of elements within the thread layer
     *
     * @tparam T_Acc alpaka accelerator [alpaka::*]
     * @param acc alpaka accelerator
     */
    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    cupla::uint3 threadDim( T_Acc const & acc )
    {
        return static_cast< uint3 >(
            ::alpaka::getWorkDiv<
                ::alpaka::Thread,
                ::alpaka::Elems
            >( acc )
        );
    }

    /** index of the thread within the block layer
     *
     * @tparam T_Acc alpaka accelerator [alpaka::*]
     * @param acc alpaka accelerator
     */
    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    cupla::uint3 threadIdx( T_Acc const & acc )
    {
        return static_cast< uint3 >(
            ::alpaka::getIdx<
                ::alpaka::Block,
                ::alpaka::Threads
            >( acc )
        );
    }

    /** index of the block within the grid layer
     *
     * @tparam T_Acc alpaka accelerator [alpaka::*]
     * @param acc alpaka accelerator
     */
    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    cupla::uint3 blockIdx( T_Acc const & acc )
    {
        return static_cast< uint3 >(
            ::alpaka::getIdx<
                ::alpaka::Grid,
                ::alpaka::Blocks
            >( acc )
        );
    }

} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
