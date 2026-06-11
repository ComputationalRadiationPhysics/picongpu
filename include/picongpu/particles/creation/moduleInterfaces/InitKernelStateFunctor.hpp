/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of InitKernelStateFunctor
     *
     * functor initialising kernelState variable
     */
    template<typename... T_KernelConfigOptions>
    struct InitKernelStateFunctor
    {
        template<typename T_KernelStateType, typename T_Index, typename... T_AdditionalData>
        HDINLINE static void init(
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_KernelStateType& kernelState,
            T_Index const additionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
