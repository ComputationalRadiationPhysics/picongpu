/* Copyright 2019 Rene Widera
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


#if defined( CUPLA_ACC_CpuOmp2Blocks  )
#   include <cupla/config/CpuOmp2Blocks.hpp>
#elif defined( CUPLA_ACC_CpuOmp2Threads  )
#   include <cupla/config/CpuOmp2Threads.hpp>
#elif defined( CUPLA_ACC_CpuSerial )
#   include <cupla/config/CpuSerial.hpp>
#elif defined( CUPLA_ACC_CpuTbbBlocks  )
#   include <cupla/config/CpuTbbBlocks.hpp>
#elif defined( CUPLA_ACC_CpuThreads  )
#   include <cupla/config/CpuThreads.hpp>
#elif defined( CUPLA_ACC_GpuCudaRt  )
#   include <cupla/config/GpuCudaRt.hpp>
#elif defined( CUPLA_ACC_GpuHipRt  )
#   include <cupla/config/GpuHipRt.hpp>
#endif

struct IncrementKernel
{
    template<typename T_Acc>
    ALPAKA_FN_ACC void operator()( T_Acc const & acc, int * ptr) const
    {
        for( int i = 0; i < elemDim.x; ++i )
            atomicAdd(ptr, 1);
    }
};


void callIncrementKernel(int* pr_d)
{
    // increment 42 times
    CUPLA_KERNEL_OPTI(
        IncrementKernel
    )(7, 6)(pr_d);
}
