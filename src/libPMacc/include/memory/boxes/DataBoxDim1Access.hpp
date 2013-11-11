/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
 

#ifndef DataBoxDim1Access_HPP
#define	DataBoxDim1Access_HPP

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"

namespace PMacc
{
namespace private_Box
{
template<unsigned DIM, class Base>
class DataBoxDim1Access;

template<class Base>
class DataBoxDim1Access<DIM1, Base> : public Base
{
public:

    enum
    {
        Dim = DIM1
    };

    HDINLINE DataBoxDim1Access(Base base, PMacc::DataSpace<Dim>) : Base(base)
    {
    }

    HDINLINE DataBoxDim1Access(PMacc::DataSpace<Dim>) : Base()
    {
    }
};

template<class Base>
class DataBoxDim1Access<DIM2, Base> : public Base
{
public:

    enum
    {
        Dim = DIM2
    };
    typedef typename Base::ValueType ValueType;
    typedef typename Base::RefValueType RefValueType;

    HDINLINE RefValueType operator()(const PMacc::DataSpace<DIM1> &idx = PMacc::DataSpace<DIM1>()) const
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx.x()));
        return (Base::operator[](real_idx.y()))[real_idx.x()];
    }

    HDINLINE RefValueType operator()(const PMacc::DataSpace<DIM1> &idx = PMacc::DataSpace<DIM1>())
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx.x()));
        return (Base::operator[](real_idx.y()))[real_idx.x()];
    }

    HDINLINE RefValueType operator[](const int idx) const
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
        return (Base::operator[](real_idx.y()))[real_idx.x()];
    }

    HDINLINE RefValueType operator[](const int idx)
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
        return (Base::operator[](real_idx.y()))[real_idx.x()];
    }

    HDINLINE DataBoxDim1Access(Base base, PMacc::DataSpace<Dim> originalSize) : Base(base), originalSize(originalSize)
    {
    }

    HDINLINE DataBoxDim1Access(PMacc::DataSpace<Dim> originalSize) : Base(), originalSize(originalSize)
    {
    }
private:
    const PMACC_ALIGN(originalSize, PMacc::DataSpace<Dim>);
};

template<class Base>
class DataBoxDim1Access<DIM3, Base> : public Base
{
public:

    enum
    {
        Dim = DIM3
    };
    typedef typename Base::ValueType ValueType;
    typedef typename Base::RefValueType RefValueType;

    HDINLINE RefValueType operator()(const PMacc::DataSpace<DIM1> &idx = PMacc::DataSpace<DIM1>()) const
    {
        const PMacc::DataSpace<DIM3> real_idx(DataSpaceOperations<DIM3>::map(originalSize, idx.x()));
        return (Base::operator[](real_idx.z()))[real_idx.y()][real_idx.x()];
    }

    HDINLINE RefValueType operator()(const PMacc::DataSpace<DIM1> &idx = PMacc::DataSpace<DIM1>())
    {
        const PMacc::DataSpace<DIM3> real_idx(DataSpaceOperations<DIM3>::map(originalSize, idx.x()));
        return (Base::operator[](real_idx.z()))[real_idx.y()][real_idx.x()];
    }

    HDINLINE RefValueType operator[](const int idx) const
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
        return (Base::operator[](real_idx.z()))[real_idx.y()][real_idx.x()];
    }

    HDINLINE RefValueType operator[](const int idx)
    {
        const PMacc::DataSpace<Dim> real_idx(DataSpaceOperations<Dim>::map(originalSize, idx));
        return (Base::operator[](real_idx.z()))[real_idx.y()][real_idx.x()];
    }

    HDINLINE DataBoxDim1Access(Base base, PMacc::DataSpace<DIM3> originalSize) : Base(base), originalSize(originalSize)
    {
    }

    HDINLINE DataBoxDim1Access(PMacc::DataSpace<DIM3> originalSize) : Base(), originalSize(originalSize)
    {
    }
private:
    const PMACC_ALIGN(originalSize, PMacc::DataSpace<DIM3>);
};
}

template<class Base>
class DataBoxDim1Access : public private_Box::DataBoxDim1Access<Base::Dim, Base>
{
public:

    typedef typename Base::ValueType ValueType;
    typedef DataBoxDim1Access<Base> Type;

    HDINLINE DataBoxDim1Access(Base base, DataSpace<Base::Dim> size) : private_Box::DataBoxDim1Access<Base::Dim, Base>(base, size)
    {
    }

    HDINLINE DataBoxDim1Access(DataSpace<Base::Dim> size) : private_Box::DataBoxDim1Access<Base::Dim, Base>(size)
    {
    }

};

}

#endif	/* DataBoxDim1Access_HPP */

