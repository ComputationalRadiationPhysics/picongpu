/**
 * Copyright 2013 Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
#ifndef ALGORITHM_KERNEL_DETAIL_SPHERICMAPPER_HPP
#define ALGORITHM_KERNEL_DETAIL_SPHERICMAPPER_HPP

#include "types.h"
#include "math/vector/Size_t.hpp"

namespace PMacc
{
namespace algorithm
{
namespace kernel
{
namespace detail
{

template<int dim, typename BlockDim>
class SphericMapper;

template<typename BlockDim>
class SphericMapper<1, BlockDim>
{
public:
    static const int dim = 1;
    SphericMapper(math::Size_t<1>) {}
    
    dim3 cudaGridDim(const math::Size_t<1>& size) const
    {
        return dim3(size.x() / BlockDim::x::value, 1, 1);
    }

    HDINLINE
    math::Int<1> operator()(const math::Int<1>& _blockIdx,
                              const math::Int<1>& _threadIdx) const
    {
        return _blockIdx.x() * BlockDim::x::value + _threadIdx.x();
    }

    HDINLINE
    math::Int<1> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<1>(_blockIdx.x),
                          math::Int<1>(_threadIdx.x));
    }
};

template<typename BlockDim>
class SphericMapper<2, BlockDim>
{
public:
    static const int dim = 2;

    SphericMapper(math::Size_t<2>) {}

    dim3 cudaGridDim(const math::Size_t<2>& size) const
    {
        return dim3(size.x() / BlockDim::x::value,
                    size.y() / BlockDim::y::value, 1);
    }

    HDINLINE
    math::Int<2> operator()(const math::Int<2>& _blockIdx,
                              const math::Int<2>& _threadIdx) const
    {
        return math::Int<2>( _blockIdx.x() * BlockDim::x::value + _threadIdx.x(),
                             _blockIdx.y() * BlockDim::y::value + _threadIdx.y() );
    }

    HDINLINE
    math::Int<2> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<2>(_blockIdx.x, _blockIdx.y),
                          math::Int<2>(_threadIdx.x, _threadIdx.y));
    }
};

template<typename BlockDim>
class SphericMapper<3, BlockDim>
{
private:
    int widthInBlocks;
public:
    static const int dim = 3;

    SphericMapper(const math::Size_t<3>& size)
     : widthInBlocks(size.x() / BlockDim::x::value) {}

    dim3 cudaGridDim(const math::Size_t<3>& size) const
    {
        return dim3(size.x() / BlockDim::x::value,
                    size.y() / BlockDim::y::value,
                    size.z() / BlockDim::z::value);
    }

    HDINLINE
    math::Int<3> operator()(const math::Int<3>& _blockIdx,
                             const math::Int<3>& _threadIdx) const
    {
        return math::Int<3>( _blockIdx * (math::Int<3>)BlockDim().vec() + _threadIdx );
    }

    HDINLINE
    math::Int<3> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<3>(_blockIdx.x, _blockIdx.y, _blockIdx.z),
                          math::Int<3>(_threadIdx.x, _threadIdx.y, _threadIdx.z));
    }
};
    
} // detail
} // kernel
} // algorithm
} // PMacc

#endif // ALGORITHM_KERNEL_DETAIL_SPHERICMAPPER_HPP
