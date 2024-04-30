/* Copyright 2023 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <array>
#include <string>

namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * @brief Describes a particle property (name, units) and how to calculate/get this property from the particle
         */
        template<typename T_Quantity, typename T_Functor>
        class FunctorDescription
        {
        public:
            using QuantityType = T_Quantity;
            using FunctorType = T_Functor;

            /** Functor which access the particle property */
            FunctorType functor;
            /** String used in the OpenPMD output */
            std::string name;
            /** The dimensionality of the particle property (defaults to dimensionless) */
            std::array<double, numUnits> units;
            FunctorDescription(
                const FunctorType func,
                std::string label,
                const std::array<double, numUnits> uDimension)
                : functor{func}
                , name{label}
                , units{uDimension}
            {
            }
        };

        /**
         */

        /**
         * @brief Describes the functors, units and names for the axes and the deposited quantity
         * @todo infer T_Quantity from T_Functor, needs particle type also, different particles may have different
         * return types
         * @tparam QuantityType The type returned by the functor
         * @tparam FunctorType Automatically deduced type of the functor
         * @param functor Functor which access the particle property
         * @param name Name for the functor/axis written out with the openPMD output.
         * @param units The dimensionality of the quantity returned by the functor in the 7D format. Defaults to
         * unitless.
         * @returns FunctorDescription object
         */
        template<typename QuantityType, typename FunctorType>
        HINLINE auto createFunctorDescription(
            FunctorType functor,
            std::string name,
            std::array<double, numUnits> units = std::array<double, numUnits>({0., 0., 0., 0., 0., 0., 0.}))
        {
            return FunctorDescription<QuantityType, FunctorType>(functor, name, units);
        }

    } // namespace plugins::binning
} // namespace picongpu
