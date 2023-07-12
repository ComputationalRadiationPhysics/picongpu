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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalElectronHistogramOverSubscribedField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalFoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalRejectionProbabilityCacheField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeStepField.hpp"
#include "picongpu/particles/atomicPhysics2/stage/CreateLocalRateCacheField.hpp"

#include <pmacc/meta/ForEach.hpp>

namespace picongpu::particles::atomicPhysics2
{
    struct AtomicPhysicsSuperCellFields
    {
        using AtomicPhysicsSpecies =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;

        //! create all superCell fields required by the atomicPhysics core loops, are stored in dataConnector
        HINLINE static void create(DataConnector& dataConnector, picongpu::MappingDesc const mappingDesc)
        {
            // local electron interaction histograms
            auto localSuperCellElectronHistogramField = std::make_unique<electronDistribution::LocalHistogramField<
                // defined/set in atomicPhysics2.param
                picongpu::atomicPhysics2::ElectronHistogram,
                // defined in memory.param
                picongpu::MappingDesc>>(mappingDesc, "Electron");
            dataConnector.consume(std::move(localSuperCellElectronHistogramField));

            ///@todo repeat for "Photons" once implemented, Brian Marre, 2022

            // local rate cache, create in pre-stage call for each species
            pmacc::meta::ForEach<
                AtomicPhysicsSpecies,
                particles::atomicPhysics2::stage::CreateLocalRateCacheField<boost::mpl::_1>>
                ForEachIonSpeciesCreateLocalRateCacheField;
            ForEachIonSpeciesCreateLocalRateCacheField(dataConnector, mappingDesc);

            // local time remaining field
            auto localSuperCellTimeRemainingField
                = std::make_unique<localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(localSuperCellTimeRemainingField));

            // local time step field
            auto localSuperCellTimeStepField
                = std::make_unique<localHelperFields::LocalTimeStepField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(localSuperCellTimeStepField));

            // local electron histogram over subscribed switch
            auto localSuperCellElectronHistogramOverSubscribedField = std::make_unique<
                localHelperFields::LocalElectronHistogramOverSubscribedField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(localSuperCellElectronHistogramOverSubscribedField));

            // local storage for FoundUnboundIonField
            auto localFoundUnboundIonField
                = std::make_unique<localHelperFields::LocalFoundUnboundIonField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(localFoundUnboundIonField));

            // rejection probability for each over-subscribed bin of the local electron histogram
            auto localSuperCellRejectionProbabilityCacheField
                = std::make_unique<localHelperFields::LocalRejectionProbabilityCacheField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(localSuperCellRejectionProbabilityCacheField));
        }
    };
} // namespace picongpu::particles::atomicPhysics2
