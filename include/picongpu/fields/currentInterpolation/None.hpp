/* Copyright 2015-2021 Axel Huebl, Benjamin Worpitz, Sergei Bastrakov
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

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu
{
    namespace fields
    {
        namespace currentInterpolation
        {
            /* None interpolated current assignment functor
             *
             * Default for staggered grids/Yee-scheme.
             * Updates field E only.
             */
            struct None
            {
                static constexpr uint32_t dim = simDim;

                using LowerMargin = typename pmacc::math::CT::make_Int<dim, 0>::type;
                using UpperMargin = LowerMargin;

                template<typename T_DataBoxE, typename T_DataBoxB, typename T_DataBoxJ>
                HDINLINE void operator()(T_DataBoxE fieldE, T_DataBoxB const, T_DataBoxJ const fieldJ)
                {
                    DataSpace<dim> const self;

                    constexpr float_X deltaT = DELTA_T;
                    fieldE(self) -= fieldJ(self) * (float_X(1.0) / EPS0) * deltaT;
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "none");
                    return propList;
                }
            };

        } // namespace currentInterpolation
    } // namespace fields

    namespace traits
    {
        /* Get margin of the None current interpolation
         *
         * This class defines a LowerMargin and an UpperMargin.
         */
        template<>
        struct GetMargin<fields::currentInterpolation::None>
        {
        private:
            using MyInterpolation = fields::currentInterpolation::None;

        public:
            using LowerMargin = typename MyInterpolation::LowerMargin;
            using UpperMargin = typename MyInterpolation::UpperMargin;
        };

    } // namespace traits
} // namespace picongpu
