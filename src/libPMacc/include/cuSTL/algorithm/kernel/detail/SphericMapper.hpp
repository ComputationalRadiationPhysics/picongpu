/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#pragma once

#include "math/vector/Size_t.hpp"
#include "types.h"

#include <boost/mpl/void.hpp>

namespace PMacc
{
namespace algorithm
{
namespace kernel
{
namespace detail
{

namespace mpl = boost::mpl;

/** The SphericMapper maps from cuda blockIdx and/or threadIdx to the cell index
 * \tparam dim dimension
 * \tparam BlockSize compile-time vector of the cuda block size (optional)
 * \tparam dummy neccesary to implement the optional BlockSize parameter
 *
 * If BlockSize is given the cuda variable blockDim is not used which is faster.
 */
template<int dim, typename BlockSize = mpl::void_, typename dummy = mpl::void_>
struct SphericMapper;

/* Compile-time BlockSize */

template<typename BlockSize>
struct SphericMapper<1, BlockSize>
{
    BOOST_STATIC_CONSTEXPR int dim = 1;

    dim3 cudaGridDim(const math::Size_t<1>& size) const
    {
        return dim3(size.x() / BlockSize::x::value, 1, 1);
    }

    HDINLINE
    math::Int<1> operator()(const math::Int<1>& _blockIdx,
                              const math::Int<1>& _threadIdx) const
    {
        return _blockIdx.x() * BlockSize::x::value + _threadIdx.x();
    }

    HDINLINE
    math::Int<1> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<1>((int)_blockIdx.x),
                          math::Int<1>((int)_threadIdx.x));
    }
};

template<typename BlockSize>
struct SphericMapper<2, BlockSize>
{
    BOOST_STATIC_CONSTEXPR int dim = 2;

    dim3 cudaGridDim(const math::Size_t<2>& size) const
    {
        return dim3(size.x() / BlockSize::x::value,
                    size.y() / BlockSize::y::value, 1);
    }

    HDINLINE
    math::Int<2> operator()(const math::Int<2>& _blockIdx,
                              const math::Int<2>& _threadIdx) const
    {
        return math::Int<2>( _blockIdx.x() * BlockSize::x::value + _threadIdx.x(),
                             _blockIdx.y() * BlockSize::y::value + _threadIdx.y() );
    }

    HDINLINE
    math::Int<2> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<2>(_blockIdx.x, _blockIdx.y),
                          math::Int<2>(_threadIdx.x, _threadIdx.y));
    }
};

template<typename BlockSize>
struct SphericMapper<3, BlockSize>
{
    BOOST_STATIC_CONSTEXPR int dim = 3;

    dim3 cudaGridDim(const math::Size_t<3>& size) const
    {
        return dim3(size.x() / BlockSize::x::value,
                    size.y() / BlockSize::y::value,
                    size.z() / BlockSize::z::value);
    }

    HDINLINE
    math::Int<3> operator()(const math::Int<3>& _blockIdx,
                             const math::Int<3>& _threadIdx) const
    {
        return math::Int<3>( _blockIdx * (math::Int<3>)BlockSize().toRT() + _threadIdx );
    }

    HDINLINE
    math::Int<3> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<3>(_blockIdx.x, _blockIdx.y, _blockIdx.z),
                          math::Int<3>(_threadIdx.x, _threadIdx.y, _threadIdx.z));
    }
};

/* Runtime BlockSize */

template<>
struct SphericMapper<1, mpl::void_>
{
    BOOST_STATIC_CONSTEXPR int dim = 1;

    dim3 cudaGridDim(const math::Size_t<1>& size, const math::Size_t<3>& blockDim) const
    {
        return dim3(size.x() / blockDim.x(), 1, 1);
    }

    DINLINE
    math::Int<1> operator()(const math::Int<1>& _blockIdx,
                              const math::Int<1>& _threadIdx) const
    {
        return _blockIdx.x() * blockDim.x + _threadIdx.x();
    }

    DINLINE
    math::Int<1> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<1>((int)_blockIdx.x),
                          math::Int<1>((int)_threadIdx.x));
    }
};

template<>
struct SphericMapper<2, mpl::void_>
{
    BOOST_STATIC_CONSTEXPR int dim = 2;

    dim3 cudaGridDim(const math::Size_t<2>& size, const math::Size_t<3>& blockDim) const
    {
        return dim3(size.x() / blockDim.x(),
                    size.y() / blockDim.y(), 1);
    }

    DINLINE
    math::Int<2> operator()(const math::Int<2>& _blockIdx,
                              const math::Int<2>& _threadIdx) const
    {
        return math::Int<2>( _blockIdx.x() * blockDim.x + _threadIdx.x(),
                             _blockIdx.y() * blockDim.y + _threadIdx.y() );
    }

    DINLINE
    math::Int<2> operator()(const dim3& _blockIdx, const dim3& _threadIdx = dim3(0,0,0)) const
    {
        return operator()(math::Int<2>(_blockIdx.x, _blockIdx.y),
                          math::Int<2>(_threadIdx.x, _threadIdx.y));
    }
};

template<>
struct SphericMapper<3, mpl::void_>
{
    BOOST_STATIC_CONSTEXPR int dim = 3;

    dim3 cudaGridDim(const math::Size_t<3>& size, const math::Size_t<3>& blockDim) const
    {
        return dim3(size.x() / blockDim.x(),
                    size.y() / blockDim.y(),
                    size.z() / blockDim.z());
    }

    DINLINE
    math::Int<3> operator()(const math::Int<3>& _blockIdx,
                             const math::Int<3>& _threadIdx) const
    {
        return math::Int<3>( _blockIdx.x() * blockDim.x + _threadIdx.x(),
                             _blockIdx.y() * blockDim.y + _threadIdx.y(),
                             _blockIdx.z() * blockDim.z + _threadIdx.z() );
    }

    DINLINE
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
