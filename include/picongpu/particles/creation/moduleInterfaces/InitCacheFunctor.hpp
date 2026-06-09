/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Brian Marre
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
