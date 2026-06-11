/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interface of CollectiveInitFunctor
     *
     * functor handling collective init of a cached data box, for example field caches
     */
    template<typename T_Index, typename... T_KernelConfigOptions>
    struct InitCacheFunctor
    {
        template<typename T_Worker, typename... T_AdditionalData>
        HDINLINE static auto getCache(
            T_Worker worker,
            T_Index const superCellIndex,
            T_AdditionalData&&... additonalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
