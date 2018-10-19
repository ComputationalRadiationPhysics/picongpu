/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "picongpu/simulation_defines.hpp"
#include <pmacc/types.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "picongpu/algorithms/FieldToParticleInterpolation.hpp"
#include "picongpu/algorithms/ShiftCoordinateSystem.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <pmacc/compileTime/AllCombinations.hpp>

namespace picongpu
{
namespace currentSolverZigZag
{
using namespace pmacc;


/** calculate the assignment factor
 *
 * Calculate assignment factor of the particle shape depending on the distance
 * between the middle point of the particle and a grid point.
 *
 * @tparam T_Shape assignment shape of a particle
 * @tparam T_pos grid position
 */
template<typename T_Shape, typename T_pos>
struct EvalAssignmentFunction
{
    using ParticleAssign = typename T_Shape::ChargeAssignmentOnSupport;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        ParticleAssign shape;
        const float_X gridPos=T_pos::value;
        return shape(gridPos-parPos);
    }
};

/* all optimizations only allowed because we know that zigzag use ShiftCoordinateSystem
 *
 * for a given support the definition area of position is:
 * - Even Support: parPos [0.0;1.0)
 * - Odd Support:  parPos [-0.5;0.5)
 */
template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::P4S, bmpl::integral_c<int, 0> >
{
    typedef picongpu::particles::shapes::P4S ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {

        return ParticleAssign::ff_1st_radius(parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::P4S, bmpl::integral_c<int, 1> >
{
    typedef picongpu::particles::shapes::P4S ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        return ParticleAssign::ff_2nd_radius(float_X(1.0)-parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::P4S, bmpl::integral_c<int, -1> >
{
    typedef picongpu::particles::shapes::P4S ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(float_X(-1.0)-parPos));
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::P4S, bmpl::integral_c<int, 2> >
{
    typedef picongpu::particles::shapes::P4S ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        return ParticleAssign::ff_3rd_radius(float_X(2.0)-parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::P4S, bmpl::integral_c<int, -2> >
{
    typedef picongpu::particles::shapes::P4S ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        return ParticleAssign::ff_3rd_radius(algorithms::math::abs(float_X(-2.0)-parPos));
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::TSC, bmpl::integral_c<int, 0> >
{
    typedef picongpu::particles::shapes::TSC ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (0 - parPos)
         * |delta| < 1/2 -> means delta^2 == (-parPos)^2 is always equal to parPos^2*/
        return ParticleAssign::ff_1st_radius(parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::TSC, bmpl::integral_c<int, 1> >
{
    typedef picongpu::particles::shapes::TSC ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (1 - parPos) -> is always positive and 0 <= delta < 3/2
         * we need no abs() @see TSC shape definition */
        return ParticleAssign::ff_2nd_radius(float_X(1.0)-parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::TSC, bmpl::integral_c<int, -1> >
{
    typedef picongpu::particles::shapes::TSC ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (-1 - parPos) -> can be negative but |delta|<3/2 */
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(float_X(-1.0)-parPos));
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::PCS, bmpl::integral_c<int, 0> >
{
    typedef picongpu::particles::shapes::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (0 - parPos) -> always negative and |delta|<1
         * |delta| == |-parPos| == |parPos|*/
        return ParticleAssign::ff_1st_radius(algorithms::math::abs(parPos));
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::PCS, bmpl::integral_c<int, 1> >
{
    typedef picongpu::particles::shapes::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (1 - parPos) -> always positive and |delta|<1
         * |delta| == |1-parPos| == 1-parPos*/
        return ParticleAssign::ff_1st_radius(float_X(1.0)-parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::PCS, bmpl::integral_c<int, 2> >
{
    typedef picongpu::particles::shapes::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (2 - parPos) -> always positive and 1<=|delta|<2
         * |delta| == |2-parPos| == 2-parPos*/
        return ParticleAssign::ff_2nd_radius(float_X(2.0)-parPos);
    }
};

template<>
struct EvalAssignmentFunction<picongpu::particles::shapes::PCS, bmpl::integral_c<int, -1> >
{
    typedef picongpu::particles::shapes::PCS ParticleAssign;

    HDINLINE float_X
    operator()(const float_X parPos)
    {
        /* delta = (-1 - parPos) -> always negative and 1<=|delta|<2
         * |delta| == |-1-parPos|*/
        return ParticleAssign::ff_2nd_radius(algorithms::math::abs(float_X(-1.0)-parPos));
    }
};

} //namespace currentSolverZigZag

} //namespace picongpu
