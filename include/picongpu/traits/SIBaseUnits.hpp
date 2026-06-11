/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace picongpu
{
    /* openPMD uses the powers of the 7 SI base measures to describe
     * the unit of a record
     * @see http://git.io/vROmP */
    constexpr uint32_t NUnitDimension = 7;

    // pre-C++11 "scoped enumerator" work-around
    namespace SIBaseUnits
    {
        enum SIBaseUnits_t
        {
            /** L */
            length = 0,
            /** M */
            mass = 1,
            /** T */
            time = 2,
            /** I */
            electricCurrent = 3,
            /** theta */
            thermodynamicTemperature = 4,
            /** N */
            amountOfSubstance = 5,
            /** J */
            luminousIntensity = 6,
        };
    } // namespace SIBaseUnits

} // namespace picongpu
