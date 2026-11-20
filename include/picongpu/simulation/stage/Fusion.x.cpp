/* Copyright 2019-2025 Rene Widera, Pawel Ordyna, Filip Optolowicz
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


#include "picongpu/simulation/stage/Fusion.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/fusion/fusion.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/particleToGrid/FoldDeriveFields.hpp"
#include "picongpu/particles/particleToGrid/combinedAttributes/CombinedAttributes.def"
#include "picongpu/particles/particleToGrid/combinedAttributes/CombinedAttributes.hpp"
#include "picongpu/unitless/checkpoints.unitless"

#include <pmacc/Environment.hpp>
#include <pmacc/algorithms/GlobalReduce.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/memory/boxes/DataBoxUnaryTransform.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <cstdint>
#include <iostream>
#include <utility>

namespace picongpu::simulation::stage
{
    namespace fusion
    {
        //! "For each" implementation for calling each collider with a loop index
        struct CallColliders
        {
            template<std::size_t... I>
            HINLINE void operator()(
                std::index_sequence<I...>,
                std::shared_ptr<DeviceHeap> const& deviceHeap,
                uint32_t const& currentStep)
            {
                (particles::fusion::CallCollider<pmacc::mp_at_c<particles::fusion::FusionPipeline, I>, I>{}(
                     deviceHeap,
                     currentStep),
                 ...);
            }
        };
    } // namespace fusion

    void Fusion::operator()(MappingDesc const cellDescription, uint32_t const currentStep) const
    {
        // Call all colliders
        constexpr size_t numColliders = pmacc::mp_size<particles::fusion::FusionPipeline>::value;
        std::make_index_sequence<numColliders> indexColliders{};
        fusion::CallColliders{}(indexColliders, m_heap, currentStep);
    }
} // namespace picongpu::simulation::stage
