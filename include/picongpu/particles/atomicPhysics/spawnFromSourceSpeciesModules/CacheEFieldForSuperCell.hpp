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

// need simDim from dim.param
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/EFieldCache.hpp"
#include "picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/math/operation/Assign.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/SharedBox.hpp>

namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
{
    namespace s_interfaces = picongpu::particles::creation::moduleInterfaces;

    //! definition of Modul
    template<typename T_IPDModel, typename T_fieldIonizationActive>
    struct CacheEFieldForSuperCell
        : public s_interfaces::
              InitCacheFunctor<pmacc::DataSpace<picongpu::simDim>, T_IPDModel, T_fieldIonizationActive>
    {
        //! @attention this is a collective method, needs a thread synchronize before first access of cache values
        template<
            typename T_Worker,
            typename T_LocalTimeRemainingBox,
            typename T_FoundUnboundIonBox,
            typename T_ChargeStateDataBox,
            typename T_AtomicStateDataBox,
            typename T_IPDIonizationStateDataBox,
            typename T_EFieldDataBox,
            typename... T_IPDInputBoxes>
        HDINLINE static auto getCache(
            T_Worker const& worker,
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_LocalTimeRemainingBox const,
            T_FoundUnboundIonBox const,
            T_ChargeStateDataBox const,
            T_AtomicStateDataBox const,
            T_IPDIonizationStateDataBox const,
            T_EFieldDataBox const eFieldBox,
            T_IPDInputBoxes const...)
        {
            if constexpr(T_fieldIonizationActive::value)
            {
                return EFieldCache::get(worker, superCellIndex, eFieldBox);
            }
            else
            {
                return 0._X;
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
