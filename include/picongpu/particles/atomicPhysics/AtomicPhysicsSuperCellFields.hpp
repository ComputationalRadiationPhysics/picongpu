/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUseCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCacheField_Bin.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCacheField_Cell.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/SharedResourcesOverSubscribedField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeStepField.hpp"
#include "picongpu/particles/atomicPhysics/stage/CreateRateCacheField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/meta/ForEach.hpp>

namespace picongpu::particles::atomicPhysics
{
    struct AtomicPhysicsSuperCellFields
    {
        using ListAtomicPhysicsSpecies = particles::atomicPhysics::traits::
            FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::Ion>;

        //! create all superCell fields required by the atomicPhysics core loops, are stored in dataConnector
        HINLINE static void create(DataConnector& dataConnector, picongpu::MappingDesc const mappingDesc)
        {
            // local electron interaction histograms
            auto localSuperCellElectronHistogramField = std::make_unique<electronDistribution::LocalHistogramField<
                // defined/set in atomicPhysics.param
                picongpu::atomicPhysics::ElectronHistogram,
                // defined in memory.param
                picongpu::MappingDesc>>(mappingDesc, "Electron");
            dataConnector.consume(std::move(localSuperCellElectronHistogramField));

            ///@todo repeat for "Photons" once implemented, Brian Marre, 2022

            // local rate cache, create in pre-stage call for each species
            pmacc::meta::ForEach<
                ListAtomicPhysicsSpecies,
                particles::atomicPhysics::stage::CreateRateCacheField<boost::mpl::_1>>
                ForEachIonSpeciesCreateRateCacheField;
            ForEachIonSpeciesCreateRateCacheField(dataConnector, mappingDesc);

            // local time remaining field
            auto superCellTimeRemainingField
                = std::make_unique<localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(superCellTimeRemainingField));

            // local time step field
            auto superCellTimeStepField
                = std::make_unique<localHelperFields::TimeStepField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(superCellTimeStepField));

            // local electron histogram over subscribed switch
            auto superCellSharedResourcesOverSubscribedField
                = std::make_unique<localHelperFields::SharedResourcesOverSubscribedField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(superCellSharedResourcesOverSubscribedField));

            // local storage for foundUnboundIon switch
            auto foundUnboundIonField
                = std::make_unique<localHelperFields::FoundUnboundIonField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(foundUnboundIonField));

            // local rejection probability for each over-subscribed cell
            auto superCellRejectionProbabilityCacheField_Cell
                = std::make_unique<localHelperFields::RejectionProbabilityCacheField_Cell<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(superCellRejectionProbabilityCacheField_Cell));

            // local rejection probability for each over-subscribed electron histogram bin
            auto superCellRejectionProbabilityCacheField_Bin
                = std::make_unique<localHelperFields::RejectionProbabilityCacheField_Bin<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(superCellRejectionProbabilityCacheField_Bin));

            // local field energy use cache
            auto superCellFieldEnergyUseCacheField
                = std::make_unique<localHelperFields::FieldEnergyUseCacheField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(superCellFieldEnergyUseCacheField));
        }
    };
} // namespace picongpu::particles::atomicPhysics
