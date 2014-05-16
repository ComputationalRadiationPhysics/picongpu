/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#ifndef ASSIGNER_DEVICEMEMASSIGNER_HPP
#define ASSIGNER_DEVICEMEMASSIGNER_HPP

#include <stdint.h>
#include "cuSTL/cursor/BufferCursor.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include "math/vector/Size_t.hpp"
#include "cuSTL/algorithm/kernel/run-time/Foreach.hpp"
#include "lambda/Expression.hpp"
#include "math/vector/compile-time/Int.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include "types.h"
#include <boost/math/common_factor.hpp>
#include <cassert>

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
    static void assign(Type* data, const math::Size_t<dim-1>& pitch, const Type& value,
                       const math::Size_t<dim>& size)
    {

        using namespace math;
        cursor::BufferCursor<Type, dim> cursor(data, pitch);
        zone::SphericZone<dim> myZone(size);

        /* The greatest common divisor of each component of the volume size
         * and a certain power of two value gives the best suitable block size
         */
        boost::math::gcd_evaluator<size_t> gcd; // greatest common divisor
        math::Size_t<3> blockDim(1);
        int maxValues[] = {16, 16, 4}; // maximum values for each dimension
        for(int i = 0; i < dim; i++)
            blockDim[i] = gcd(size[i], maxValues[dim-1]);
        /* the maximum number of threads per block for devices with
         * compute capability > 2.0 is 1024 */
        assert(blockDim.productOfComponents()<=1024);

        using namespace lambda;
        algorithm::kernel::RT::Foreach foreach(blockDim);
        foreach(myZone, cursor, _1 = value);
    }
};

} // assigner
} // PMacc

#endif // ASSIGNER_DEVICEMEMASSIGNER_HPP
