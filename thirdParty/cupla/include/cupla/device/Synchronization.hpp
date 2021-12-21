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

#include "cupla/types.hpp"

#include <alpaka/alpaka.hpp>

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
inline namespace device
{

    /** synchronize threads within the block
     *
     * @tparam T_Acc alpaka accelerator [alpaka::*]
     * @param acc alpaka accelerator
     *
     * @{
     */
    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    void syncThreads( T_Acc const & acc )
    {
        ::alpaka::syncBlockThreads( acc );
    }

    template< typename T_Acc >
    ALPAKA_FN_ACC ALPAKA_FN_INLINE
    void __syncthreads( T_Acc const & acc )
    {
        syncThreads( acc );
    }

    //!@}

} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
