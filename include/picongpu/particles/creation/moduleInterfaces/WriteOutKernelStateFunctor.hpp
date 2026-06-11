/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of WriteOutKernelStateFunctor
     *
     * post processing functor for writing out information computed from the kernelState to additionalData
     *
     * @details is called once for each superCell by the master thread
     */
    template<typename... T_KernelConfigOptions>
    struct WriteOutKernelStateFunctor
    {
        template<typename T_Index, typename T_KernelStateType, typename... T_AdditionalData>
        HDINLINE static void postProcess(
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_KernelStateType const kernelState,
            T_Index const additionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
