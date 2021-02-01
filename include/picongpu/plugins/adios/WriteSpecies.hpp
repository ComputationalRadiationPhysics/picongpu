/* Copyright 2014-2021 Rene Widera, Felix Schmitt, Axel Huebl,
 *                     Alexander Grund
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/adios/ADIOSWriter.def"
#include <pmacc/assert.hpp>

#include "picongpu/plugins/ISimulationPlugin.hpp"

#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/plugins/adios/writer/ParticleAttribute.hpp"

#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/operations/ConcatListOfFrames.hpp>
#include <pmacc/particles/particleFilter/FilterFactory.hpp>
#include <pmacc/particles/particleFilter/PositionFilter.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_same.hpp>


namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;

        /** Write copy particle to host memory and dump to ADIOS file
         *
         * @tparam T_Species type of species
         *
         */
        template<typename T_SpeciesFilter>
        struct WriteSpecies
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

            template<typename Space>
            HINLINE void operator()(ThreadParams* params, const Space particleOffset)
            {
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) write species: %1%") % T_SpeciesFilter::getName();
                DataConnector& dc = Environment<>::get().DataConnector();
                /* load particle without copy particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(ThisSpecies::FrameType::getName(), true);

                /* count total number of particles on the device */
                log<picLog::INPUT_OUTPUT>("ADIOS:   (begin) count particles: %1%") % T_SpeciesFilter::getName();
                // enforce that the filter interface is fulfilled
                particles::filter::IUnary<typename T_SpeciesFilter::Filter> particleFilter{params->currentStep};
                uint64_cu totalNumParticles = 0;
                totalNumParticles = pmacc::CountParticles::countOnDevice<CORE + BORDER>(
                    *speciesTmp,
                    *(params->cellDescription),
                    params->localWindowToDomainOffset,
                    params->window.localDimensions.size,
                    particleFilter);
                log<picLog::INPUT_OUTPUT>("ADIOS:   ( end ) count particles: %1% = %2%") % T_SpeciesFilter::getName()
                    % totalNumParticles;

                AdiosFrameType hostFrame;

                /* malloc host memory */
                log<picLog::INPUT_OUTPUT>("ADIOS:   (begin) malloc host memory: %1%") % T_SpeciesFilter::getName();
                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, MallocHostMemory<bmpl::_1>> mallocMem;
                mallocMem(hostFrame, totalNumParticles);
                log<picLog::INPUT_OUTPUT>("ADIOS:   ( end ) malloc host memory: %1%") % T_SpeciesFilter::getName();

                if(totalNumParticles > 0)
                {
                    log<picLog::INPUT_OUTPUT>(
                        "ADIOS:   (begin) copy particle host (with hierarchy) to host (without hierarchy): %1%")
                        % T_SpeciesFilter::getName();
                    typedef bmpl::vector<typename GetPositionFilter<simDim>::type> usedFilters;
                    typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
                    MyParticleFilter filter;
                    /* activate filter pipeline if moving window is activated */
                    filter.setStatus(MovingWindow::getInstance().isEnabled());
                    filter.setWindowPosition(params->localWindowToDomainOffset, params->window.localDimensions.size);

                    DataConnector& dc = Environment<>::get().DataConnector();

                    auto mallocMCBuffer
                        = dc.get<MallocMCBuffer<DeviceHeap>>(MallocMCBuffer<DeviceHeap>::getName(), true);

                    int globalParticleOffset = 0;
                    AreaMapping<CORE + BORDER, MappingDesc> mapper(*(params->cellDescription));

                    pmacc::particles::operations::ConcatListOfFrames<simDim> concatListOfFrames(mapper.getGridDim());

#if(PMACC_CUDA_ENABLED == 1 || ALPAKA_ACC_GPU_HIP_ENABLED == 1)
                    auto particlesBox = speciesTmp->getHostParticlesBox(mallocMCBuffer->getOffset());
#else
                    /* This separate code path is only a workaround until MallocMCBuffer
                     * is alpaka compatible.
                     *
                     * @todo remove this workaround: we know that we are allowed to access the
                     * device memory directly.
                     */
                    auto particlesBox = speciesTmp->getDeviceParticlesBox();
                    /* Notify to the event system that the particles box is used on the host.
                     *
                     * @todo remove this workaround
                     */
                    __startOperation(ITask::TASK_HOST);

#endif
                    concatListOfFrames(
                        globalParticleOffset,
                        hostFrame,
                        particlesBox,
                        filter,
                        particleOffset, /*relative to data domain (not to physical domain)*/
                        totalCellIdx_,
                        mapper,
                        particleFilter);

                    dc.releaseData(MallocMCBuffer<DeviceHeap>::getName());

                    /* this costs a little bit of time but adios writing is slower */
                    PMACC_ASSERT((uint64_cu) globalParticleOffset == totalNumParticles);
                }
                /* dump to adios file */
                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, adios::ParticleAttribute<bmpl::_1>> writeToAdios;
                writeToAdios(params, hostFrame, totalNumParticles);

                /* free host memory */
                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, FreeHostMemory<bmpl::_1>> freeMem;
                freeMem(hostFrame);
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) writing species: %1%") % T_SpeciesFilter::getName();

                /* write species counter table to adios file */
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) writing particle index table for %1%")
                    % T_SpeciesFilter::getName();
                {
                    GridController<simDim>& gc = Environment<simDim>::get().GridController();

                    const size_t pos_offset = 2;

                    /* particlesMetaInfo = (num particles, scalar position, particle offset x, y, z) */
                    uint64_t particlesMetaInfo[5] = {totalNumParticles, gc.getScalarPosition(), 0, 0, 0};
                    for(size_t d = 0; d < simDim; ++d)
                        particlesMetaInfo[pos_offset + d] = particleOffset[d];

                    /* prevent that top (y) gpus have negative value here */
                    if(gc.getPosition().y() == 0)
                        particlesMetaInfo[pos_offset + 1] = 0;

                    if(particleOffset[1] < 0) // 1 == y
                        particlesMetaInfo[pos_offset + 1] = 0;

                    int64_t adiosIndexVarId = *(params->adiosSpeciesIndexVarIds.begin());
                    params->adiosSpeciesIndexVarIds.pop_front();
                    ADIOS_CMD(adios_write_byid(params->adiosFileHandle, adiosIndexVarId, particlesMetaInfo));
                }
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) writing particle index table for %1%")
                    % T_SpeciesFilter::getName();
            }
        };


    } // namespace adios

} // namespace picongpu
