/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"

namespace PMacc
{

template<class T_Base>
class DataBoxDim1Access : protected T_Base
{
public:

    typedef T_Base Base;
    BOOST_STATIC_CONSTEXPR uint32_t Dim= Base::Dim;


    typedef typename Base::ValueType ValueType;
    typedef typename Base::RefValueType RefValueType;


    HDINLINE RefValueType operator()(const PMacc::DataSpace<DIM1> &idx = PMacc::DataSpace<DIM1>()) const
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx.x()));
        return Base::operator()(real_idx);
    }

    HDINLINE RefValueType operator()(const PMacc::DataSpace<DIM1> &idx = PMacc::DataSpace<DIM1>())
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx.x()));
        return Base::operator()(real_idx);
    }

    HDINLINE RefValueType operator[](const int idx) const
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
        return Base::operator()(real_idx);
    }

    HDINLINE RefValueType operator[](const int idx)
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
        return Base::operator()(real_idx);
    }

    HDINLINE DataBoxDim1Access(const Base base, const PMacc::DataSpace<Dim> originalSize) : Base(base), originalSize(originalSize)
    {
    }

    HDINLINE DataBoxDim1Access(const PMacc::DataSpace<Dim> originalSize) : Base(), originalSize(originalSize)
    {
    }
private:
    const PMACC_ALIGN(originalSize, PMacc::DataSpace<Dim>);

};

} //namespace
