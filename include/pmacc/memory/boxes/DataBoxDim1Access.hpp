/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"

namespace pmacc
{
    template<class T_Base>
    class DataBoxDim1Access : protected T_Base
    {
    public:
        typedef T_Base Base;
        static constexpr uint32_t Dim = Base::Dim;


        typedef typename Base::ValueType ValueType;
        typedef typename Base::RefValueType RefValueType;


        HDINLINE RefValueType operator()(const pmacc::DataSpace<DIM1>& idx = pmacc::DataSpace<DIM1>()) const
        {
            const pmacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx.x()));
            return Base::operator()(real_idx);
        }

        HDINLINE RefValueType operator()(const pmacc::DataSpace<DIM1>& idx = pmacc::DataSpace<DIM1>())
        {
            const pmacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx.x()));
            return Base::operator()(real_idx);
        }

        HDINLINE RefValueType operator[](const int idx) const
        {
            const pmacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
            return Base::operator()(real_idx);
        }

        HDINLINE RefValueType operator[](const int idx)
        {
            const pmacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
            return Base::operator()(real_idx);
        }

        HDINLINE DataBoxDim1Access(const Base base, const pmacc::DataSpace<Dim> originalSize)
            : Base(base)
            , originalSize(originalSize)
        {
        }

        HDINLINE DataBoxDim1Access(const pmacc::DataSpace<Dim> originalSize) : Base(), originalSize(originalSize)
        {
        }

    private:
        PMACC_ALIGN(originalSize, const pmacc::DataSpace<Dim>);
    };

} // namespace pmacc
