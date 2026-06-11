/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of SuperCellFilterFunctor
     *
     * @details functor returning whether entire superCell to should be skipped depending on additionalData or
     * superCell index
     *
     * @example skip superCell if TimeRemainingDataBox[additionalDataIndex] is > 0, (dataBox passed via additionalData)
     *
     *  @note to skip test, use empty function
     */
    template<typename... T_KernelConfigOptions>
    struct SuperCellFilterFunctor
    {
        //! true =^= skip superCell, false =^= process superCell
        template<typename T_Index, typename... T_AdditionalData>
        HDINLINE static bool skipSuperCell(
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_Index const additionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
