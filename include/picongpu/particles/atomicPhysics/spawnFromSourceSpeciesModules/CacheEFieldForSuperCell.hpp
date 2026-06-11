/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

// need simDim from dim.param
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/EFieldCache.hpp"
#include "picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
{
    namespace s_interfaces = picongpu::particles::creation::moduleInterfaces;

    //! definition of Modul
    template<uint32_t T_id, typename T_IPDModel, typename T_fieldIonizationActive>
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
            [[maybe_unused]] pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_LocalTimeRemainingBox const,
            T_FoundUnboundIonBox const,
            T_ChargeStateDataBox const,
            T_AtomicStateDataBox const,
            T_IPDIonizationStateDataBox const,
            [[maybe_unused]] T_EFieldDataBox const eFieldBox,
            T_IPDInputBoxes const...)
        {
            if constexpr(T_fieldIonizationActive::value)
            {
                return EFieldCache::get<T_id>(worker, superCellIndex, eFieldBox);
            }
            else
            {
                return 0._X;
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
