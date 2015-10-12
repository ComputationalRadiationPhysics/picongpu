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
#include <cuSTL/cursor/FunctorCursor.hpp>
#include "math/Vector.hpp"
#include "algorithms/ShiftCoordinateSystemNative.hpp"

namespace picongpu
{

/** interpolate field which are defined on a grid to a point inside of a grid
 *
 * interpolate around of a point from -AssignmentFunction::support/2 to
 * (AssignmentFunction::support+1)/2
 *
 * \tparam GridShiftMethod functor which shift coordinate system that al value are
 * located on corner
 * \tparam AssignmentFunction AssignmentFunction which is used for interpolation
 * \tparam InterpolationMethod functor for interpolation method
 */
template<class T_Shape, class InterpolationMethod>
struct FieldToParticleInterpolationNative
{
    typedef typename T_Shape::ChargeAssignment AssignmentFunction;
    BOOST_STATIC_CONSTEXPR int supp = AssignmentFunction::support;

    BOOST_STATIC_CONSTEXPR int lowerMargin = supp / 2;
    BOOST_STATIC_CONSTEXPR int upperMargin = (supp + 1) / 2;
    typedef typename PMacc::math::CT::make_Int<simDim,lowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<simDim,upperMargin>::type UpperMargin;

    template<class Cursor, class VecVector_ >
    HDINLINE float3_X operator()(Cursor field, const floatD_X& particlePos,
                                 const VecVector_ & fieldPos)
    {
        using namespace lambda;
        DECLARE_PLACEHOLDERS() // declares _1, _2, _3, ... in device code

        /**\brief:
         * The following three calls seperate the vector interpolation into three
         * independent scalar interpolations. In each call the coordinate system
         * is turned so that E_scalar does the interpolation for the z-component.
         */

        /** _1[mpl::int_<0>()] means:
         * Create a functor which returns [0] applied on the first paramter.
         * Here it is: return the x-component of the field-vector.
         * _1[mpl::int_<0>()] is equivalent to _1[0] but has no runtime cost.
         */

        BOOST_AUTO(field_x, PMacc::cursor::make_FunctorCursor(field, _1[mpl::int_ < 0 > ()]));
        floatD_X pos_tmp(particlePos);
        ShiftCoordinateSystemNative<supp>()(field_x, pos_tmp, fieldPos.x());
        float_X result_x = InterpolationMethod::template interpolate<AssignmentFunction, -lowerMargin, upperMargin > (field_x, pos_tmp);

        BOOST_AUTO(field_y, PMacc::cursor::make_FunctorCursor(field, _1[mpl::int_ < 1 > ()]));
        pos_tmp = particlePos;
        ShiftCoordinateSystemNative<supp>()(field_y, pos_tmp, fieldPos.y());
        float_X result_y = InterpolationMethod::template interpolate<AssignmentFunction, -lowerMargin, upperMargin > (field_y, pos_tmp);

        BOOST_AUTO(field_z, PMacc::cursor::make_FunctorCursor(field, _1[mpl::int_ < 2 > ()]));
        pos_tmp = particlePos;
        ShiftCoordinateSystemNative<supp>()(field_z, pos_tmp, fieldPos.z());
        float_X result_z = InterpolationMethod::template interpolate<AssignmentFunction, -lowerMargin, upperMargin > (field_z, pos_tmp);

        return float3_X(result_x, result_y, result_z);
    }

};

namespace traits
{

/*Get margin of a solver
 * class must define a LowerMargin and UpperMargin
 */
template<class AssignMethod, class InterpolationMethod>
struct GetMargin<picongpu::FieldToParticleInterpolationNative<AssignMethod, InterpolationMethod> >
{
private:
    typedef picongpu::FieldToParticleInterpolationNative< AssignMethod, InterpolationMethod> Interpolation;
public:
    typedef typename Interpolation::LowerMargin LowerMargin;
    typedef typename Interpolation::UpperMargin UpperMargin;
};

} //namespace traits

} //namespace picongpu


