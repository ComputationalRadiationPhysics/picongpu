/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Klaus Steiniger
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
#include "picongpu/algorithms/ShiftCoordinateSystemNative.hpp"

#include <pmacc/cuSTL/cursor/FunctorCursor.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/cuSTL/algorithm/functor/GetComponent.hpp>

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
        using AssignmentFunction = typename T_Shape::ChargeAssignment;
        static constexpr int supp = AssignmentFunction::support;

        static constexpr int lowerMargin = supp / 2;
        static constexpr int upperMargin = (supp + 1) / 2;
        using LowerMargin = typename pmacc::math::CT::make_Int<simDim, lowerMargin>::type;
        using UpperMargin = typename pmacc::math::CT::make_Int<simDim, upperMargin>::type;

        template<class Cursor, class VecVector_>
        HDINLINE float3_X operator()(Cursor field, const floatD_X& particlePos, const VecVector_& fieldPos)
        {
            /**\brief:
             * The following three calls seperate the vector interpolation into three
             * independent scalar interpolations. In each call the coordinate system
             * is turned so that E_scalar does the interpolation for the z-component.
             */

            auto field_x
                = pmacc::cursor::make_FunctorCursor(field, pmacc::algorithm::functor::GetComponent<float_X>(0));
            floatD_X pos_tmp(particlePos);
            ShiftCoordinateSystemNative<supp>()(field_x, pos_tmp, fieldPos.x());
            float_X result_x
                = InterpolationMethod::template interpolate<AssignmentFunction, -lowerMargin, upperMargin>(
                    field_x,
                    pos_tmp);

            auto field_y
                = pmacc::cursor::make_FunctorCursor(field, pmacc::algorithm::functor::GetComponent<float_X>(1));
            pos_tmp = particlePos;
            ShiftCoordinateSystemNative<supp>()(field_y, pos_tmp, fieldPos.y());
            float_X result_y
                = InterpolationMethod::template interpolate<AssignmentFunction, -lowerMargin, upperMargin>(
                    field_y,
                    pos_tmp);

            auto field_z
                = pmacc::cursor::make_FunctorCursor(field, pmacc::algorithm::functor::GetComponent<float_X>(2));
            pos_tmp = particlePos;
            ShiftCoordinateSystemNative<supp>()(field_z, pos_tmp, fieldPos.z());
            float_X result_z
                = InterpolationMethod::template interpolate<AssignmentFunction, -lowerMargin, upperMargin>(
                    field_z,
                    pos_tmp);

            return float3_X(result_x, result_y, result_z);
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            GetStringProperties<InterpolationMethod> propList;
            return propList;
        }
    };

    namespace traits
    {
        /*Get margin of a solver
         * class must define a LowerMargin and UpperMargin
         */
        template<class AssignMethod, class InterpolationMethod>
        struct GetMargin<picongpu::FieldToParticleInterpolationNative<AssignMethod, InterpolationMethod>>
        {
        private:
            using Interpolation = picongpu::FieldToParticleInterpolationNative<AssignMethod, InterpolationMethod>;

        public:
            using LowerMargin = typename Interpolation::LowerMargin;
            using UpperMargin = typename Interpolation::UpperMargin;
        };

    } // namespace traits

} // namespace picongpu
