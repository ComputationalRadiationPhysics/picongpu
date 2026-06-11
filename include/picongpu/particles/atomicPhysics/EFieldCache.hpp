/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

// need unit.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/math/operation/Assign.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>

namespace picongpu::particles::atomicPhysics
{
    struct EFieldCache
    {
        //! @attention collective Method, before first access of cache values a thread synchronize is required!
        template<uint32_t T_id, typename T_Worker, typename T_EFieldDataBox>
        HDINLINE static auto get(
            T_Worker const& worker,
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_EFieldDataBox const eFieldBox)
        {
            using SuperCellBlock = pmacc::SuperCellDescription<typename picongpu::SuperCellSize>;
            /// @note cache is unique for kernel call by id and dataType, and thereby shared between workers
            auto eFieldCache = CachedBox::create<T_id, typename T_EFieldDataBox::ValueType>(worker, SuperCellBlock());

            pmacc::DataSpace<picongpu::simDim> const superCellCellOffset
                = superCellIndex * picongpu::SuperCellSize::toRT();

            auto fieldEBlockToCache = eFieldBox.shift(superCellCellOffset);

            pmacc::math::operation::Assign assign;
            auto collective = makeThreadCollective<SuperCellBlock>();

            collective(worker, assign, eFieldCache, fieldEBlockToCache);

            return eFieldCache;
        }
    };
} // namespace picongpu::particles::atomicPhysics
