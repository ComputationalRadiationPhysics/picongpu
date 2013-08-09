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
 
#pragma once

#include <math/vector/navigator/PermutedNavigator.hpp>

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
                math::PermutedNavigator<Axes> >& type;
};
    
} // result_of
    
template<typename Axes, typename TVector>
HDINLINE
typename result_of::TwistVectorAxes<Axes, TVector>::type&
twistVectorAxes(TVector& vector)
{
    return reinterpret_cast<typename result_of::TwistVectorAxes<Axes, TVector>::type>(vector);
}
    
} // tools
} // math
} // PMacc
