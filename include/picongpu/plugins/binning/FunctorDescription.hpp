/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Tapish Narwal
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

#include "picongpu/plugins/binning/UnitConversion.hpp"

#include <array>
#include <string>
#include <type_traits>

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
            std::array<double, numUnits> units{0., 0., 0., 0., 0., 0., 0.};

            // Disabled for integral quantities since the dimension conversion will make things non integral
            FunctorDescription(FunctorType func, std::string label, std::array<double, numUnits> uDimension)
                requires std::is_floating_point_v<T_Quantity>
                : functor{std::move(func)}
                , name{std::move(label)}
                , units{std::move(uDimension)}
            {
            }

            FunctorDescription(FunctorType func, std::string label) : functor{std::move(func)}, name{std::move(label)}
            {
            }
        };

        /**
         * These functions are the analog of make_tuple etc and exist to avoid specifying the T_Functor, which is
         * automatically deduced with these functions
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
        inline auto createFunctorDescription(
            FunctorType functor,
            std::string const& name,
            std::array<double, numUnits> units)
        {
            return FunctorDescription<QuantityType, FunctorType>(functor, name, units);
        }

        template<typename QuantityType, typename FunctorType>
        inline auto createFunctorDescription(FunctorType functor, std::string const& name)
        {
            return FunctorDescription<QuantityType, FunctorType>(functor, name);
        }


    } // namespace plugins::binning
} // namespace picongpu
