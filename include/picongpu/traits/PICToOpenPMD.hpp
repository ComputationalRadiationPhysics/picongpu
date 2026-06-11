/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once


#include "picongpu/traits/Unit.hpp"
#include "picongpu/traits/UnitDimension.hpp"

#include <pmacc/types.hpp>

#include <string>
#include <vector>

namespace picongpu
{
    namespace traits
    {
        /** Reinterpret attributes for openPMD
         *
         * Currently, this conversion tables are used to translate the PIConGPU
         * totalCellIdx (unitless cell index) to the openPMD positionOffset (length)
         */
        template<typename T_Identifier>
        struct OpenPMDName;

        template<typename T_Identifier>
        struct OpenPMDUnit;

        template<typename T_Identifier>
        struct OpenPMDUnitDimension;

    } // namespace traits
} // namespace picongpu

#include "PICToOpenPMD.tpp"
