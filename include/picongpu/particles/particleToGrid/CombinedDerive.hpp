/* Copyright 2021-2023 Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/particleToGrid/CombinedDerive.def"
#include "picongpu/particles/particleToGrid/combinedAttributes/CombinedAttributes.hpp"

#include <string>
#include <vector>

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            template<
                typename T_BaseAttributeSolver,
                typename T_ModifierAttributeSolver,
                typename T_ModifyingOperation,
                typename T_AttributeDescription>
            HDINLINE float1_64 CombinedDeriveSolver<
                T_BaseAttributeSolver,
                T_ModifierAttributeSolver,
                T_ModifyingOperation,
                T_AttributeDescription>::getUnit() const
            {
                return T_AttributeDescription().getUnit();
            }

            template<
                typename T_BaseAttributeSolver,
                typename T_ModifierAttributeSolver,
                typename T_ModifyingOperation,
                typename T_AttributeDescription>
            HINLINE std::vector<float_64> CombinedDeriveSolver<
                T_BaseAttributeSolver,
                T_ModifierAttributeSolver,
                T_ModifyingOperation,
                T_AttributeDescription>::getUnitDimension() const
            {
                return T_AttributeDescription().getUnitDimension();
            }

            template<
                typename T_BaseAttributeSolver,
                typename T_ModifierAttributeSolver,
                typename T_ModifyingOperation,
                typename T_AttributeDescription>
            HINLINE std::string CombinedDeriveSolver<
                T_BaseAttributeSolver,
                T_ModifierAttributeSolver,
                T_ModifyingOperation,
                T_AttributeDescription>::getName()
            {
                return T_AttributeDescription::getName();
            }
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
