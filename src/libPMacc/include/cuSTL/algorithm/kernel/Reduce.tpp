/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "math/vector/Size_t.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "lambda/Expression.hpp"
#include "cuSTL/container/compile-time/SharedBuffer.hpp"
#include "math/Vector.hpp"
#include "cuSTL/cursor/NestedCursor.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include <boost/type_traits/remove_reference.hpp>
#include <cuSTL/cursor/navigator/EmptyNavigator.hpp>

namespace PMacc
{
namespace algorithm
{
namespace kernel
{

namespace detail
{

template<typename BlockDim, int dim>
struct ReduceKernel
{
    typedef void result_type;
    int width;

    HDINLINE
    ReduceKernel() {};

    HDINLINE
    ReduceKernel(int width)
     : width(width) {}

    template<typename TCursor, typename Data, typename Functor>
    DINLINE void operator()(const TCursor& resultPerBlock, const Data& data, const Functor& functor) const
    {
        if(dim == 1)
            if(blockIdx.x * BlockDim::x::value + threadIdx.x >= width) return;

        using namespace math;
        container::CT::SharedBuffer<Data, CT::Int<CT::volume<BlockDim>::type::value> > shBuffer;
        BOOST_AUTO(sh, shBuffer.origin());
        int linearThreadIdx = threadIdx.z * BlockDim::x::value * BlockDim::y::value +
                              threadIdx.y * BlockDim::x::value + threadIdx.x;

        sh[linearThreadIdx] = data;
        int numThreads;
        __shared__ bool odd;
        if(dim == 1 && blockIdx.x == (gridDim.x-1))
        {
            numThreads = width % BlockDim::x::value;
            if(numThreads == 0) numThreads = BlockDim::x::value;
            odd = numThreads % 2;
        }
        else
            numThreads = CT::volume<BlockDim>::type::value;

        numThreads /= 2;
        while(numThreads > 0)
        {
            if(linearThreadIdx >= numThreads) return;
            __syncthreads(); //\todo: durch gescheitere __syncthreads funktionen ersetzen
            Data tmp = functor(sh[2*linearThreadIdx], sh[2*linearThreadIdx+1]);
            if((dim == 1) && (blockIdx.x == (gridDim.x-1)) && (threadIdx.x == (numThreads-1)) && odd)
                tmp = functor(tmp, sh[2*linearThreadIdx+2]);

            __syncthreads();
            sh[linearThreadIdx] = tmp;

            if(dim == 1 && blockIdx.x == (gridDim.x-1) && threadIdx.x == (numThreads-1))
                odd = numThreads % 2;
            numThreads /= 2;
        }
        if(linearThreadIdx != 0) return;
        resultPerBlock[int(blockIdx.y * gridDim.x + blockIdx.x)] = *sh;
    }
};

}

template<typename BlockDim>
template<typename DestCursor, typename Zone, typename SrcCursor, typename Functor>
void Reduce<BlockDim>::operator()(const DestCursor& destCursor, const Zone& p_zone, const SrcCursor& srcCursor, const Functor& functor)
{
    typedef typename boost::remove_reference<typename DestCursor::type>::type type;

    BOOST_AUTO(_destCursor, cursor::make_Cursor(destCursor.getAccessor(),
                                                cursor::EmptyNavigator(),
                                                destCursor.getMarker()));
    container::DeviceBuffer<type, 1>* partialSum[2];
    partialSum[0] = new container::DeviceBuffer<type, 1>
        (p_zone.size.productOfComponents() / BlockDim().toRT().productOfComponents());
    partialSum[1] = new container::DeviceBuffer<type, 1>(partialSum[0]->size());
    int curDestBuffer = 0;
    int partialSumSize = partialSum[curDestBuffer]->size().x();

    using namespace lambda;

    if(partialSumSize == 1)
    {
        Foreach<BlockDim>()(p_zone, srcCursor,
            expr(detail::ReduceKernel<BlockDim, Zone::dim>(p_zone.size.x()))
                (_destCursor, _1, make_Functor(functor)));

    }
    else
    {
        Foreach<BlockDim>()(p_zone, srcCursor,
            expr(detail::ReduceKernel<BlockDim, Zone::dim>(p_zone.size.x()))
                (partialSum[curDestBuffer]->origin(), _1, make_Functor(functor)));
    }

    while(partialSumSize > 1)
    {
        int numBlocks = ceil((float)partialSumSize / 512.0f);
        zone::SphericZone<1> p_zone1D = zone::SphericZone<1>(math::Size_t<1>((size_t)(numBlocks*512)));
        curDestBuffer ^= 1;
        if(numBlocks == 1)
        {
            Foreach<math::CT::Int<512,1,1> >()(p_zone1D, partialSum[(curDestBuffer+1)%2]->origin(),
                            expr(detail::ReduceKernel<math::CT::Int<512,1,1>, 1>(partialSumSize))
                            (_destCursor, _1, make_Functor(functor)));
        }
        else
        {
            Foreach<math::CT::Int<512,1,1> >()(p_zone1D, partialSum[(curDestBuffer+1)%2]->origin(),
                            expr(detail::ReduceKernel<math::CT::Int<512,1,1>, 1>(partialSumSize))
                            (partialSum[curDestBuffer]->origin(), _1, make_Functor(functor)));
        }
        partialSumSize = numBlocks;
    }

    delete partialSum[0];
    delete partialSum[1];
}

} // kernel
} // algorithm
} // PMacc
