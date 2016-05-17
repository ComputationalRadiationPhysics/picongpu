/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
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
#include "algorithms/ShiftCoordinateSystem.hpp"

namespace picongpu
{

/** interpolate field which are defined on a grid to a point inside of a grid
 *
 * interpolate around a point from -AssignmentFunction::support/2 to
 * (AssignmentFunction::support+1)/2
 *
 * \tparam GridShiftMethod functor which shift coordinate system that al value are
 * located on corner
 * \tparam AssignmentFunction AssignmentFunction which is used for interpolation
 * \tparam InterpolationMethod functor for interpolation method
 */
template<class T_Shape, class InterpolationMethod>
struct FieldToParticleInterpolation
{
    typedef typename T_Shape::ChargeAssignmentOnSupport AssignmentFunction;
    BOOST_STATIC_CONSTEXPR int supp = AssignmentFunction::support;

    BOOST_STATIC_CONSTEXPR int lowerMargin = supp / 2 ;
    BOOST_STATIC_CONSTEXPR int upperMargin = (supp + 1) / 2;
    typedef typename PMacc::math::CT::make_Int<simDim,lowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<simDim,upperMargin>::type UpperMargin;

    /*(supp + 1) % 2 is 1 for even supports else 0*/
    BOOST_STATIC_CONSTEXPR int begin = -supp / 2 + (supp + 1) % 2;
    BOOST_STATIC_CONSTEXPR int end = begin+supp-1;

    template<class Cursor, class VecVector>
    HDINLINE typename Cursor::ValueType operator()(Cursor field,
                                                   const floatD_X& particlePos,
                                                   const VecVector& fieldPos)
    {
        using namespace lambda;
        DECLARE_PLACEHOLDERS() // declares _1, _2, _3, ... in device code

        /**\brief:
         * The following calls seperate the vector interpolation into
         * independent scalar interpolations.
         */

        /** _1[i] means:
         * Create a functor which returns [i] applied on the first paramter.
         * Here it is: return the i-component of the field-vector.
         */

        typedef typename PMacc::math::CT::make_Int<simDim,supp>::type Supports;

        typename Cursor::ValueType result;
        for(uint32_t i = 0; i < Cursor::ValueType::dim; i++)
        {
            BOOST_AUTO(fieldComponent, PMacc::cursor::make_FunctorCursor(field, _1[i]));
            floatD_X particlePosShifted = particlePos;
            ShiftCoordinateSystem<Supports>()(fieldComponent, particlePosShifted, fieldPos[i]);
            result[i] = InterpolationMethod::template interpolate<AssignmentFunction, begin, end > (fieldComponent, particlePosShifted);
        }

        return result;
    }

};

namespace traits
{

/*Get margin of a solver
 * class must define a LowerMargin and UpperMargin
 */
template<class AssignMethod, class InterpolationMethod>
struct GetMargin<picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod> >
{
private:
    typedef picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod> Interpolation;
public:
    typedef typename Interpolation::LowerMargin LowerMargin;
    typedef typename Interpolation::UpperMargin UpperMargin;
};

} //namespace traits

} //namespace picongpu


