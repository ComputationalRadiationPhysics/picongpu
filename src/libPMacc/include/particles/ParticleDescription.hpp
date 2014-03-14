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

#include <boost/mpl/vector.hpp>

namespace PMacc
{



/** ParticleDescription defines attributes, methods and flags of a particle
 * 
 * This type holds no runtime data, this is only to describe all elements of a particle
 *
 * @tparam T_ValueTypeSeq sequence with value_identifier
 * @tparam T_MethodsList sequence of classes with particle methods 
 *                       (e.g. calculate mass, gamma, ...)
 * @tparam T_Flags sequence with idenifierer to add fags on a frame 
 *                 (e.g. useSolverXY, calcRadiation, ...) 
 */
template<
typename T_ValueTypeSeq,
typename T_MethodsList = bmpl::vector<>,
typename T_Flags = bmpl::vector<> >
struct ParticleDescription
{
    typedef T_ValueTypeSeq ValueTypeSeq;
    typedef T_MethodsList MethodsList;
    typedef T_Flags FlagsList;
    typedef ParticleDescription<ValueTypeSeq, MethodsList, FlagsList> ThisType;
};

} //namespace PMacc




