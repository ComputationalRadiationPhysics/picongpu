/**
 * Copyright 2013 René Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "simulation_defines.hpp"
#include "traits/GetComponentType.hpp"

namespace picongpu
{

namespace traits
{

/** Trait for bool */
template<>
struct GetComponentType<bool>
{
    typedef bool type;
};

/** Trait for float */
template<>
struct GetComponentType<float>
{
    typedef float type;
};

/** Trait for double */
template<>
struct GetComponentType<double>
{
    typedef double type;
};

/** Trait for int */
template<>
struct GetComponentType<int>
{
    typedef int type;
};

/** Trait for float_X */
template<typename T_DataType,int T_Dim>
struct GetComponentType<pmacc::math::Vector<T_DataType,T_Dim> >
{
    typedef T_DataType type;
};

/** Trait for float_X */
template<unsigned DIM>
struct GetComponentType<DataSpace<DIM> >
{
    typedef int type;
};

} //namespace traits

}// namespace picongpu
