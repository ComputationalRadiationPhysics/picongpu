/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
