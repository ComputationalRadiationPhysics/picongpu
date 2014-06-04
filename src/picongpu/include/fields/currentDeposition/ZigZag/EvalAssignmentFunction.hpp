/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "basicOperations.hpp"
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "algorithms/FieldToParticleInterpolation.hpp"
#include "algorithms/ShiftCoordinateSystem.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>
#include "compileTime/AllCombinations.hpp"

namespace picongpu
{
namespace currentSolverZigZag
{
using namespace PMacc;



template<typename T_Shape, typename T_pos>
struct EvalAssignmentFunction
{
    typedef typename T_Shape::ChargeAssignmentOnSupport ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        ParticleAssign shape;
        return shape(delta);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::TSC, bmpl::integral_c<int, 0> >
{
    typedef typename picongpu::particleShape::TSC ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_1st_radius(delta);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::TSC, bmpl::integral_c<int, 1> >
{
    typedef typename picongpu::particleShape::TSC ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_2nd_radius(delta);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::TSC, bmpl::integral_c<int, -1 > >
{
    typedef typename picongpu::particleShape::TSC ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(delta));
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::PCS, bmpl::integral_c<int, 0> >
{
    typedef typename picongpu::particleShape::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_1st_radius(algorithms::math::abs(delta));
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::PCS, bmpl::integral_c<int, 1> >
{
    typedef typename picongpu::particleShape::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_1st_radius(delta);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::PCS, bmpl::integral_c<int, 2> >
{
    typedef typename picongpu::particleShape::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_2nd_radius(delta);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particleShape::PCS, bmpl::integral_c<int, -1 > >
{
    typedef typename picongpu::particleShape::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X delta)
    {
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(delta));
    }
};

} //namespace currentSolverZigZag

} //namespace picongpu
