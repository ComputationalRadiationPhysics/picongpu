/* Copyright 2014-2021 Felix Schmitt, Axel Huebl
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <mpi.h>

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/adios/ADIOSWriter.def"

#include "picongpu/plugins/ISimulationPlugin.hpp"

#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/particles/traits/GetSpeciesFlagName.hpp"
#include "picongpu/traits/PICToAdios.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include "picongpu/plugins/adios/writer/ParticleAttributeSize.hpp"
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits.hpp>

#include <string>


namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;


        /** Count number of particles for a species
         *
         * @tparam T_Species type of species
         *
         */
        template<typename T_SpeciesFilter>
        struct ADIOSCountParticles
        {
        public:
            typedef typename T_SpeciesFilter::Species ThisSpecies;
            typedef typename ThisSpecies::FrameType FrameType;
            typedef typename FrameType::ParticleDescription ParticleDescription;
            typedef typename FrameType::ValueTypeSeq ParticleAttributeList;

            /* delete multiMask and localCellIdx in adios particle*/
            typedef bmpl::vector<multiMask, localCellIdx> TypesToDelete;
            typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

            /* add totalCellIdx for adios particle*/
            typedef typename MakeSeq<ParticleCleanedAttributeList, totalCellIdx>::type ParticleNewAttributeList;

            typedef typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type
                NewParticleDescription;

            typedef Frame<OperatorCreateVectorBox, NewParticleDescription> AdiosFrameType;

            HINLINE void operator()(ThreadParams* params)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                uint64_t mpiSize = gc.getGlobalSize();
                uint64_t mpiRank = gc.getGlobalRank();

                const std::string speciesGroup(T_SpeciesFilter::getName() + "/");
                const std::string speciesPath(
                    params->adiosBasePath + std::string(ADIOS_PATH_PARTICLES) + speciesGroup);

                /* load particle without copy particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(ThisSpecies::FrameType::getName(), true);
                // enforce that the filter interface is fulfilled
                particles::filter::IUnary<typename T_SpeciesFilter::Filter> particleFilter{params->currentStep};
                /* count total number of particles on the device */
                uint64_cu totalNumParticles = 0;
                totalNumParticles = pmacc::CountParticles::countOnDevice<CORE + BORDER>(
                    *speciesTmp,
                    *(params->cellDescription),
                    params->localWindowToDomainOffset,
                    params->window.localDimensions.size,
                    particleFilter);

                /* MPI_Allgather to compute global size and my offset */
                uint64_t myNumParticles = totalNumParticles;
                uint64_t allNumParticles[mpiSize];
                uint64_t globalNumParticles = 0;
                uint64_t myParticleOffset = 0;

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    &myNumParticles,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    allNumParticles,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    gc.getCommunicator().getMPIComm()));

                for(uint64_t i = 0; i < mpiSize; ++i)
                {
                    globalNumParticles += allNumParticles[i];
                    if(i < mpiRank)
                        myParticleOffset += allNumParticles[i];
                }

                /* iterate over all attributes of this species */
                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, adios::ParticleAttributeSize<bmpl::_1>>
                    attributeSize;
                attributeSize(params, speciesGroup, myNumParticles, globalNumParticles, myParticleOffset);

                /* TODO: constant particle records */

                /* openPMD ED-PIC: additional attributes */
                traits::PICToAdios<float_64> adiosDoubleType;
                const float_64 particleShape(GetShape<ThisSpecies>::type::assignmentFunctionOrder);
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "particleShape",
                    speciesPath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &particleShape));

                traits::GetSpeciesFlagName<ThisSpecies, current<>> currentDepositionName;
                const std::string currentDeposition(currentDepositionName());
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "currentDeposition",
                    speciesPath.c_str(),
                    adios_string,
                    1,
                    (void*) currentDeposition.c_str()));

                traits::GetSpeciesFlagName<ThisSpecies, particlePusher<>> particlePushName;
                const std::string particlePush(particlePushName());
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "particlePush",
                    speciesPath.c_str(),
                    adios_string,
                    1,
                    (void*) particlePush.c_str()));

                traits::GetSpeciesFlagName<ThisSpecies, interpolation<>> particleInterpolationName;
                const std::string particleInterpolation(particleInterpolationName());
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "particleInterpolation",
                    speciesPath.c_str(),
                    adios_string,
                    1,
                    (void*) particleInterpolation.c_str()));

                const std::string particleSmoothing("none");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "particleSmoothing",
                    speciesPath.c_str(),
                    adios_string,
                    1,
                    (void*) particleSmoothing.c_str()));

                /* define adios var for species index/info table */
                {
                    const uint64_t localTableSize = 5;
                    traits::PICToAdios<uint64_t> adiosIndexType;

                    const char* path = nullptr;
                    int64_t adiosSpeciesIndexVar = defineAdiosVar<DIM1>(
                        params->adiosGroupHandle,
                        (speciesPath + "particles_info").c_str(),
                        path,
                        adiosIndexType.type,
                        pmacc::math::UInt64<DIM1>(localTableSize),
                        pmacc::math::UInt64<DIM1>(localTableSize * uint64_t(gc.getGlobalSize())),
                        pmacc::math::UInt64<DIM1>(localTableSize * uint64_t(gc.getGlobalRank())),
                        true,
                        params->adiosCompression);

                    params->adiosSpeciesIndexVarIds.push_back(adiosSpeciesIndexVar);

                    params->adiosGroupSize += sizeof(uint64_t) * localTableSize * gc.getGlobalSize();
                }
            }
        };


    } // namespace adios

} // namespace picongpu
