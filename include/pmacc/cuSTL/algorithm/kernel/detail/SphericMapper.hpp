/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/void.hpp>

namespace pmacc
{
    namespace algorithm
    {
        namespace kernel
        {
            namespace detail
            {
                namespace mpl = boost::mpl;

                /** The SphericMapper maps from cupla blockIdx and/or threadIdx to the cell index
                 * \tparam dim dimension
                 * \tparam BlockSize compile-time vector of the cupla block size (optional)
                 * \tparam dummy neccesary to implement the optional BlockSize parameter
                 *
                 * If BlockSize is given the cupla variable blockDim is not used which is faster.
                 */
                template<int dim, typename BlockSize = mpl::void_, typename dummy = mpl::void_>
                struct SphericMapper;

                /* Compile-time BlockSize */

                template<typename BlockSize>
                struct SphericMapper<1, BlockSize>
                {
                    static constexpr int dim = 1;

                    typename math::Size_t<3>::BaseType cuplaGridDim(const math::Size_t<1>& size) const
                    {
                        return math::Size_t<3>(size.x() / BlockSize::x::value, 1u, 1u);
                    }

                    template<typename T_Acc>
                    HDINLINE math::Int<1> operator()(
                        T_Acc const& acc,
                        const math::Int<1>& _blockIdx,
                        const math::Int<1>& _threadIdx) const
                    {
                        return _blockIdx.x() * BlockSize::x::value + _threadIdx.x();
                    }

                    template<typename T_Acc>
                    HDINLINE math::Int<1> operator()(
                        T_Acc const& acc,
                        const cupla::dim3& _blockIdx,
                        const cupla::dim3& _threadIdx = cupla::dim3(0, 0, 0)) const
                    {
                        return operator()(acc, math::Int<1>((int) _blockIdx.x), math::Int<1>((int) _threadIdx.x));
                    }
                };

                template<typename BlockSize>
                struct SphericMapper<2, BlockSize>
                {
                    static constexpr int dim = 2;

                    typename math::Size_t<3>::BaseType cuplaGridDim(const math::Size_t<2>& size) const
                    {
                        return math::Size_t<3>(size.x() / BlockSize::x::value, size.y() / BlockSize::y::value, 1u);
                    }

                    template<typename T_Acc>
                    HDINLINE math::Int<2> operator()(
                        T_Acc const& acc,
                        const math::Int<2>& _blockIdx,
                        const math::Int<2>& _threadIdx) const
                    {
                        return math::Int<2>(
                            _blockIdx.x() * BlockSize::x::value + _threadIdx.x(),
                            _blockIdx.y() * BlockSize::y::value + _threadIdx.y());
                    }

                    template<typename T_Acc>
                    HDINLINE math::Int<2> operator()(
                        T_Acc const& acc,
                        const cupla::dim3& _blockIdx,
                        const cupla::dim3& _threadIdx = cupla::dim3(0, 0, 0)) const
                    {
                        return operator()(
                            acc,
                            math::Int<2>(_blockIdx.x, _blockIdx.y),
                            math::Int<2>(_threadIdx.x, _threadIdx.y));
                    }
                };

                template<typename BlockSize>
                struct SphericMapper<3, BlockSize>
                {
                    static constexpr int dim = 3;

                    typename math::Size_t<3>::BaseType cuplaGridDim(const math::Size_t<3>& size) const
                    {
                        return math::Size_t<3>(
                            size.x() / BlockSize::x::value,
                            size.y() / BlockSize::y::value,
                            size.z() / BlockSize::z::value);
                    }

                    template<typename T_Acc>
                    HDINLINE math::Int<3> operator()(
                        T_Acc const& acc,
                        const math::Int<3>& _blockIdx,
                        const math::Int<3>& _threadIdx) const
                    {
                        return math::Int<3>(_blockIdx * (math::Int<3>) BlockSize().toRT() + _threadIdx);
                    }

                    template<typename T_Acc>
                    HDINLINE math::Int<3> operator()(
                        T_Acc const& acc,
                        const cupla::dim3& _blockIdx,
                        const cupla::dim3& _threadIdx = cupla::dim3(0, 0, 0)) const
                    {
                        return operator()(
                            acc,
                            math::Int<3>(_blockIdx.x, _blockIdx.y, _blockIdx.z),
                            math::Int<3>(_threadIdx.x, _threadIdx.y, _threadIdx.z));
                    }
                };

                /* Runtime BlockSize */

                template<>
                struct SphericMapper<1, mpl::void_>
                {
                    static constexpr int dim = 1;

                    typename math::Size_t<3>::BaseType cuplaGridDim(
                        const math::Size_t<1>& size,
                        const math::Size_t<3>& blockSize) const
                    {
                        return math::Size_t<3>(size.x() / blockSize.x(), 1u, 1u);
                    }

                    template<typename T_Acc>
                    DINLINE math::Int<1> operator()(
                        T_Acc const& acc,
                        const math::Int<1>& _blockDim,
                        const math::Int<1>& _blockIdx,
                        const math::Int<1>& _threadIdx) const
                    {
                        return _blockIdx.x() * _blockDim.x() + _threadIdx.x();
                    }

                    template<typename T_Acc>
                    DINLINE math::Int<1> operator()(
                        T_Acc const& acc,
                        const cupla::dim3& _blockDim,
                        const cupla::dim3& _blockIdx,
                        const cupla::dim3& _threadIdx) const
                    {
                        return operator()(
                            acc,
                            math::Int<1>((int) _blockDim.x),
                            math::Int<1>((int) _blockIdx.x),
                            math::Int<1>((int) _threadIdx.x));
                    }
                };

                template<>
                struct SphericMapper<2, mpl::void_>
                {
                    static constexpr int dim = 2;

                    typename math::Size_t<3>::BaseType cuplaGridDim(
                        const math::Size_t<2>& size,
                        const math::Size_t<3>& blockSize) const
                    {
                        return math::Size_t<3>(size.x() / blockSize.x(), size.y() / blockSize.y(), 1);
                    }

                    template<typename T_Acc>
                    DINLINE math::Int<2> operator()(
                        T_Acc const& acc,
                        const math::Int<2>& _blockDim,
                        const math::Int<2>& _blockIdx,
                        const math::Int<2>& _threadIdx) const
                    {
                        return math::Int<2>(
                            _blockIdx.x() * _blockDim.x() + _threadIdx.x(),
                            _blockIdx.y() * _blockDim.y() + _threadIdx.y());
                    }

                    template<typename T_Acc>
                    DINLINE math::Int<2> operator()(
                        T_Acc const& acc,
                        const cupla::dim3& _blockDim,
                        const cupla::dim3& _blockIdx,
                        const cupla::dim3& _threadIdx) const
                    {
                        return operator()(
                            acc,
                            math::Int<2>(_blockDim.x, _blockDim.y),
                            math::Int<2>(_blockIdx.x, _blockIdx.y),
                            math::Int<2>(_threadIdx.x, _threadIdx.y));
                    }
                };

                template<>
                struct SphericMapper<3, mpl::void_>
                {
                    static constexpr int dim = 3;

                    typename math::Size_t<3>::BaseType cuplaGridDim(
                        const math::Size_t<3>& size,
                        const math::Size_t<3>& blockSize) const
                    {
                        return math::Size_t<3>(
                            size.x() / blockSize.x(),
                            size.y() / blockSize.y(),
                            size.z() / blockSize.z());
                    }

                    template<typename T_Acc>
                    DINLINE math::Int<3> operator()(
                        T_Acc const& acc,
                        const math::Int<3>& _blockDim,
                        const math::Int<3>& _blockIdx,
                        const math::Int<3>& _threadIdx) const
                    {
                        return math::Int<3>(
                            _blockIdx.x() * _blockDim.x() + _threadIdx.x(),
                            _blockIdx.y() * _blockDim.y() + _threadIdx.y(),
                            _blockIdx.z() * _blockDim.z() + _threadIdx.z());
                    }

                    template<typename T_Acc>
                    DINLINE math::Int<3> operator()(
                        T_Acc const& acc,
                        const cupla::dim3& _blockDim,
                        const cupla::dim3& _blockIdx,
                        const cupla::dim3& _threadIdx) const
                    {
                        return operator()(
                            acc,
                            math::Int<3>(_blockDim.x, _blockDim.y, _blockDim.z),
                            math::Int<3>(_blockIdx.x, _blockIdx.y, _blockIdx.z),
                            math::Int<3>(_threadIdx.x, _threadIdx.y, _threadIdx.z));
                    }
                };

            } // namespace detail
        } // namespace kernel
    } // namespace algorithm
} // namespace pmacc
