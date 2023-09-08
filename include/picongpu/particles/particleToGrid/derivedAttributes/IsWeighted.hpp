/* Copyright 2022-2023 Sergei Bastrakov
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

#include <type_traits>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                /** Derive attribute trait whether its value is scaled to macroparticle weight
                 *
                 * Note that it describes how the derived attribute itself is calculated, not the eventual field.
                 * Logically it roughly corresponds to openPMD macroWeighted = true and weightingPower = 1.0
                 * @see traits::MacroWeighted @see traits::WeightingPower.
                 * However, as derived attributes are a separate quantity from species we have a separate trait.
                 *
                 * Inherits std::true_type, std::false_type or a compatible type.
                 *
                 * @tparam T_DerivedAttribute derived attribute type
                 */
                template<typename T_DerivedAttribute>
                struct IsWeighted : public std::false_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
