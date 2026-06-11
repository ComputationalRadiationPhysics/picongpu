/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of AdditionalDataIndexFunctor
     *
     * functor returning index to access additionalData by depending on the areaMapping and the superCellIdx
     *
     * @note only one is supported for all additionalData
     * @note may be ignored for some or all additionalData
     */
    template<typename... T_KernelConfigOptions>
    struct AdditionalDataIndexFunctor
    {
        //! may be overwritten by implementation
        static constexpr uint8_t indexDim = picongpu::simDim;

        template<typename T_AreaMapping>
        HDINLINE static pmacc::DataSpace<indexDim> getIndex(
            T_AreaMapping const areaMapping,
            pmacc::DataSpace<picongpu::simDim> const superCellIdx);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
