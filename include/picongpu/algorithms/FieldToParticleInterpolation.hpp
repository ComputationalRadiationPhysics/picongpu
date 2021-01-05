/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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
#include <pmacc/cuSTL/cursor/FunctorCursor.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/cuSTL/algorithm/functor/GetComponent.hpp>
#include "picongpu/algorithms/ShiftCoordinateSystem.hpp"

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
        using AssignmentFunction = typename T_Shape::ChargeAssignmentOnSupport;
        static constexpr int supp = AssignmentFunction::support;

        static constexpr int lowerMargin = supp / 2;
        static constexpr int upperMargin = (supp + 1) / 2;
        using LowerMargin = typename pmacc::math::CT::make_Int<simDim, lowerMargin>::type;
        using UpperMargin = typename pmacc::math::CT::make_Int<simDim, upperMargin>::type;

        PMACC_CASSERT_MSG(
            __FieldToParticleInterpolation_supercell_is_too_small_for_stencil,
            pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                    >= lowerMargin
                && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                    >= upperMargin);

        /*(supp + 1) % 2 is 1 for even supports else 0*/
        static constexpr int begin = -supp / 2 + (supp + 1) % 2;
        static constexpr int end = begin + supp - 1;


        template<class Cursor, class VecVector>
        HDINLINE typename Cursor::ValueType operator()(
            Cursor field,
            const floatD_X& particlePos,
            const VecVector& fieldPos)
        {
            /**\brief:
             * The following calls seperate the vector interpolation into
             * independent scalar interpolations.
             */
            using Supports = typename pmacc::math::CT::make_Int<simDim, supp>::type;

            typename Cursor::ValueType result;
            for(uint32_t i = 0; i < Cursor::ValueType::dim; i++)
            {
                auto fieldComponent
                    = pmacc::cursor::make_FunctorCursor(field, pmacc::algorithm::functor::GetComponent<float_X>(i));
                floatD_X particlePosShifted = particlePos;
                ShiftCoordinateSystem<Supports>()(fieldComponent, particlePosShifted, fieldPos[i]);
                result[i] = InterpolationMethod::template interpolate<AssignmentFunction, begin, end>(
                    fieldComponent,
                    particlePosShifted);
            }

            return result;
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
        struct GetMargin<picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod>>
        {
        private:
            using Interpolation = picongpu::FieldToParticleInterpolation<AssignMethod, InterpolationMethod>;

        public:
            using LowerMargin = typename Interpolation::LowerMargin;
            using UpperMargin = typename Interpolation::UpperMargin;
        };

    } // namespace traits

} // namespace picongpu
