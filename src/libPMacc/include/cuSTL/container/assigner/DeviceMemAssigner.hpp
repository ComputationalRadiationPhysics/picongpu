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
 
#ifndef ASSIGNER_DEVICEMEMASSIGNER_HPP
#define ASSIGNER_DEVICEMEMASSIGNER_HPP

#include <stdint.h>
#include "cuSTL/cursor/BufferCursor.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include "math/vector/Size_t.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "lambda/Expression.hpp"
#include "math/vector/compile-time/Int.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include "types.h"

namespace PMacc
{
namespace assigner
{
    
namespace mpl = boost::mpl;
    
template<int _dim>
struct DeviceMemAssigner
{
    static const int dim = _dim;
    template<typename Type>
    HDINLINE
    static void assign(Type* data, const math::Size_t<dim-1>& pitch, const Type& value,
                       const math::Size_t<dim>& size)
    {
#ifndef __CUDA_ARCH__
        using namespace math;
        cursor::BufferCursor<Type, dim> cursor(data, pitch);
        zone::SphericZone<dim> myZone(size);
        
        typedef typename mpl::vector<math::CT::Int<256,1,1>, 
                                     math::CT::Int<16,16,1>, 
                                     math::CT::Int<8,8,4> >::type BlockDims;
        using namespace lambda;
        if(size == Size_t<dim>(1))
            algorithm::kernel::Foreach<math::CT::Int<1,1,1> >()(myZone, cursor, _1 = value);
        else
            algorithm::kernel::Foreach<typename mpl::at_c<BlockDims, dim-1>::type>()(myZone, cursor, _1 = value);
#endif
    }
};
    
} // assigner
} // PMacc

#endif // ASSIGNER_DEVICEMEMASSIGNER_HPP
