/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it and or modify
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

#include "picongpu/defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of SuperCellFilterFunctor
     *
     * @details functor returning whether entire superCell to should be skipped depending on additionalData or
     * superCell index
     *
     * @example skip superCell if localTimeRemainingDataBox[additionalDataIndex](dataBox passed via additionalData) is
     * > 0
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
