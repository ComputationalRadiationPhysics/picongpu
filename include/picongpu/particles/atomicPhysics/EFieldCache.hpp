/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
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

// need unit.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics
{
    struct EFieldCache
    {
        //! @attention collective Method, before first access of cache values a thread synchronize is required!
        template<typename T_Worker, typename T_EFieldDataBox>
        HDINLINE static auto get(
            T_Worker const& worker,
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_EFieldDataBox const eFieldBox)
        {
            using SuperCellBlock = pmacc::SuperCellDescription<typename picongpu::SuperCellSize>;
            /// @note cache is unique for kernel call by id and dataType, and thereby shared between workers
            auto eFieldCache
                = CachedBox::create<__COUNTER__, typename T_EFieldDataBox::ValueType::type>(worker, SuperCellBlock());

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
