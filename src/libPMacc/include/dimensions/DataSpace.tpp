/**
 * Copyright 2013 Rene Widera
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


#pragma once

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"

namespace PMacc
{

namespace traits
{

template<unsigned DIM>
struct GetComponentsType<DataSpace<DIM>, false >
{
    typedef typename DataSpace<DIM>::type type;
};

/** Trait for float_X */
template<unsigned DIM>
struct GetNComponents<DataSpace<DIM>,false >
{
    static const uint32_t value=DIM;
};

}// namespace traits

} //namespace PMacc
