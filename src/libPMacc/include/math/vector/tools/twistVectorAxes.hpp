/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include <math/vector/navigator/PermutedNavigator.hpp>
#include <math/vector/navigator/StackedNavigator.hpp>

namespace PMacc
{
namespace math
{
namespace tools
{
    
namespace result_of
{
    
template<typename Axes, typename TVector>
struct TwistVectorAxes
{
    typedef math::Vector<typename TVector::type, TVector::dim, typename TVector::Accessor,
            math::StackedNavigator<typename TVector::Navigator, math::PermutedNavigator<Axes> > >& type;
};
    
} // result_of
    
/** Returns a reference of vector with twisted axes.
 * 
 * The axes twist operation is done in place. This means that the result refers to the 
 * memory of the input vector. The input vector's navigator policy is replaced by 
 * a new navigator which merely consists of the old navigator plus a twisting navigator.
 * This new navigator does not use any real memory.
 */
template<typename Axes, typename TVector>
HDINLINE
typename result_of::TwistVectorAxes<Axes, TVector>::type
twistVectorAxes(TVector& vector)
{
    /* The reinterpret_cast is valid because the target type is the same than the
     * input type except its navigator policy which does not occupy any memory though.
     */
    return reinterpret_cast<typename result_of::TwistVectorAxes<Axes, TVector>::type>(vector);
}
    
} // tools
} // math
} // PMacc
