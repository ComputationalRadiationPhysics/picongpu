/*
 * SPDX-FileCopyrightText: Brian Marre, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/particles/atomicPhysics/debug/param.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <memory>
#include <stdexcept>

namespace picongpu::particles::atomicPhysics::stage
{
    /** pre-simulation stage initiating the rateCacheField for atomicPhysics
     *
     * is a stage to
     * @tparam T_IonSpecies species for which to call the functor
     */
    template<typename T_IonSpecies>
    struct CreateRateCacheField
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        template<typename T_MappingDescription>
        HINLINE void operator()(DataConnector& dataConnector, T_MappingDescription const& mappingDesc) const
        {
            auto rateCacheField = std::make_unique<picongpu::particles::atomicPhysics::localHelperFields::
                                                       RateCacheField<picongpu::MappingDesc, IonSpecies>>(mappingDesc);
            dataConnector.consume(std::move(rateCacheField));
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
