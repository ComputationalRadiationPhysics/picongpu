/* Copyright 2014-2021 Rene Widera, Felix Schmitt, Axel Huebl,
 *                     Alexander Grund, Franz Poeschel
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
#include "picongpu/particles/traits/GetSpeciesFlagName.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/kernel/CopySpecies.kernel"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/plugins/openPMD/openPMDVersion.def"
#include "picongpu/plugins/openPMD/writer/ParticleAttribute.hpp"
#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/plugins/output/ConstSpeciesAttributes.hpp"
#include "picongpu/plugins/openPMD/openPMDDimension.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/eventSystem/events/kernelEvents.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/operations/ConcatListOfFrames.hpp>
#include <pmacc/particles/particleFilter/FilterFactory.hpp>
#include <pmacc/particles/particleFilter/PositionFilter.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>


#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_same.hpp>


namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        template<typename SpeciesTmp, typename Filter, typename ParticleFilter, typename ParticleOffset>
        struct StrategyRunParameters
        {
            pmacc::DataConnector& dc;
            ThreadParams& params;
            SpeciesTmp& speciesTmp;
            Filter& filter;
            ParticleFilter& particleFilter;
            ParticleOffset& particleOffset;
            uint64_t myNumParticles, globalNumParticles;
            StrategyRunParameters(
                pmacc::DataConnector& c_dc,
                ThreadParams& c_params,
                SpeciesTmp& c_speciesTmp,
                Filter& c_filter,
                ParticleFilter& c_particleFilter,
                ParticleOffset& c_particleOffset,
                uint64_t c_myNumParticles,
                uint64_t c_globalNumParticles)
                : dc(c_dc)
                , params(c_params)
                , speciesTmp(c_speciesTmp)
                , filter(c_filter)
                , particleFilter(c_particleFilter)
                , particleOffset(c_particleOffset)
                , myNumParticles(c_globalNumParticles)
                , globalNumParticles(c_globalNumParticles)
            {
            }
        };

        template<typename openPMDFrameType, typename RunParameters>
        struct Strategy
        {
            virtual void malloc(std::string name, openPMDFrameType&, uint64_cu const myNumParticles) = 0;

            virtual void free(openPMDFrameType& hostFrame) = 0;

            virtual void prepare(std::string name, openPMDFrameType& hostFrame, RunParameters) = 0;

            virtual ~Strategy() = default;
        };

        /*
         * Use double buffer.
         */
        template<typename openPMDFrameType, typename RunParameters>
        struct StrategyADIOS : Strategy<openPMDFrameType, RunParameters>
        {
            void malloc(std::string name, openPMDFrameType& hostFrame, uint64_cu const myNumParticles) override
            {
                /* malloc host memory */
                log<picLog::INPUT_OUTPUT>("openPMD:   (begin) malloc host memory: %1%") % name;
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, MallocHostMemory<bmpl::_1>> mallocMem;
                mallocMem(hostFrame, myNumParticles);
                log<picLog::INPUT_OUTPUT>("openPMD:   ( end ) malloc host memory: %1%") % name;
            }

            void free(openPMDFrameType& hostFrame) override
            {
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, FreeHostMemory<bmpl::_1>> freeMem;
                freeMem(hostFrame);
            }


            void prepare(std::string name, openPMDFrameType& hostFrame, RunParameters rp) override
            {
                log<picLog::INPUT_OUTPUT>("openPMD:   (begin) copy particle host (with hierarchy) to "
                                          "host (without hierarchy): %1%")
                    % name;
                auto mallocMCBuffer
                    = rp.dc.template get<MallocMCBuffer<DeviceHeap>>(MallocMCBuffer<DeviceHeap>::getName(), true);

                int globalParticleOffset = 0;
                AreaMapping<CORE + BORDER, MappingDesc> mapper(*(rp.params.cellDescription));

                pmacc::particles::operations::ConcatListOfFrames<simDim> concatListOfFrames(mapper.getGridDim());

#if(PMACC_CUDA_ENABLED == 1 || ALPAKA_ACC_GPU_HIP_ENABLED == 1)
                auto particlesBox = rp.speciesTmp->getHostParticlesBox(mallocMCBuffer->getOffset());
#else
                /* This separate code path is only a workaround until
                 * MallocMCBuffer is alpaka compatible.
                 *
                 * @todo remove this workaround: we know that we are allowed to
                 * access the device memory directly.
                 */
                auto particlesBox = rp.speciesTmp->getDeviceParticlesBox();
                /* Notify to the event system that the particles box is used on
                 * the host.
                 *
                 * @todo remove this workaround
                 */
                __startOperation(ITask::TASK_HOST);

#endif
                concatListOfFrames(
                    globalParticleOffset,
                    hostFrame,
                    particlesBox,
                    rp.filter,
                    rp.particleOffset, /*relative to data domain (not to physical
                                       domain)*/
                    totalCellIdx_,
                    mapper,
                    rp.particleFilter);

                rp.dc.releaseData(MallocMCBuffer<DeviceHeap>::getName());

                /* this costs a little bit of time but writing to external is
                 * slower in general */
                PMACC_ASSERT((uint64_cu) globalParticleOffset == rp.globalNumParticles);
            }
        };

        /*
         * Use mapped memory.
         */
        template<typename openPMDFrameType, typename RunParameters>
        struct StrategyHDF5 : Strategy<openPMDFrameType, RunParameters>
        {
            void malloc(std::string name, openPMDFrameType& hostFrame, uint64_cu const myNumParticles) override
            {
                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) malloc mapped memory: %1%") % name;
                /*malloc mapped memory*/
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, MallocMemory<bmpl::_1>> mallocMem;
                mallocMem(hostFrame, myNumParticles);
                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) malloc mapped memory: %1%") % name;
            }

            void free(openPMDFrameType& hostFrame) override
            {
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, FreeMemory<bmpl::_1>> freeMem;
                freeMem(hostFrame);
            }

            void prepare(std::string name, openPMDFrameType& hostFrame, RunParameters rp) override
            {
                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) copy particle to host: %1%") % name;

                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) get mapped memory device pointer: %1%") % name;
                /*load device pointer of mapped memory*/
                openPMDFrameType deviceFrame;
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1>> getDevicePtr;
                getDevicePtr(deviceFrame, hostFrame);
                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) get mapped memory device pointer: %1%") % name;

                GridBuffer<int, DIM1> counterBuffer(DataSpace<DIM1>(1));
                AreaMapping<CORE + BORDER, MappingDesc> mapper(*(rp.params.cellDescription));

                constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                /* this sanity check costs a little bit of time but hdf5 writing is
                 * slower */
                PMACC_KERNEL(CopySpecies<numWorkers>{})
                (mapper.getGridDim(), numWorkers)(
                    counterBuffer.getDeviceBuffer().getPointer(),
                    deviceFrame,
                    rp.speciesTmp->getDeviceParticlesBox(),
                    rp.filter,
                    rp.particleOffset,
                    totalCellIdx_,
                    mapper,
                    rp.particleFilter);
                counterBuffer.deviceToHost();
                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) copy particle to host: %1%") % name;
                __getTransactionEvent().waitForFinished();
                log<picLog::INPUT_OUTPUT>("openPMD:  all events are finished: %1%") % name;

                PMACC_ASSERT((uint64_t) counterBuffer.getHostBuffer().getDataBox()[0] == rp.myNumParticles);
            }
        };

        /** Write copy particle to host memory and dump to openPMD file
         *
         * @tparam T_Species type of species
         */
        template<typename T_SpeciesFilter, typename T_Species = T_SpeciesFilter>
        struct WriteSpecies
        {
        public:
            using ThisSpecies = typename T_SpeciesFilter::Species;
            using FrameType = typename ThisSpecies::FrameType;
            using ParticleDescription = typename FrameType::ParticleDescription;
            using ParticleAttributeList = typename FrameType::ValueTypeSeq;

            /* delete multiMask and localCellIdx in openPMD particle*/
            using TypesToDelete = bmpl::vector<multiMask, localCellIdx>;
            using ParticleCleanedAttributeList = typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type;

            /* add totalCellIdx for openPMD particle*/
            using ParticleNewAttributeList = typename MakeSeq<ParticleCleanedAttributeList, totalCellIdx>::type;

            using NewParticleDescription =
                typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type;

            using openPMDFrameType = Frame<OperatorCreateVectorBox, NewParticleDescription>;

            void setParticleAttributes(
                ::openPMD::ParticleSpecies& record,
                AbstractJsonMatcher& matcher,
                std::string const& basename)
            {
                const float_64 particleShape(GetShape<ThisSpecies>::type::assignmentFunctionOrder);
                record.setAttribute("particleShape", particleShape);

                traits::GetSpeciesFlagName<ThisSpecies, current<>> currentDepositionName;
                const std::string currentDeposition(currentDepositionName());
                record.setAttribute("currentDeposition", currentDeposition.c_str());

                traits::GetSpeciesFlagName<ThisSpecies, particlePusher<>> particlePushName;
                const std::string particlePush(particlePushName());
                record.setAttribute("particlePush", particlePush.c_str());

                traits::GetSpeciesFlagName<ThisSpecies, interpolation<>> particleInterpolationName;
                const std::string particleInterpolation(particleInterpolationName());
                record.setAttribute("particleInterpolation", particleInterpolation.c_str());

                const std::string particleSmoothing("none");
                record.setAttribute("particleSmoothing", particleSmoothing.c_str());

                // now we have a map in a writeable format with all zeroes
                // for each record copy it and modify the copy, e.g.

                // const records stuff
                ::openPMD::Datatype dataType = ::openPMD::Datatype::DOUBLE;
                ::openPMD::Extent extent = {0};
                ::openPMD::Dataset dataSet = ::openPMD::Dataset(dataType, extent);

                // mass
                plugins::output::GetMassOrZero<FrameType> const getMassOrZero;
                if(getMassOrZero.hasMassRatio)
                {
                    const float_64 mass(getMassOrZero());
                    auto& massRecord = record["mass"];
                    auto& massComponent = massRecord[::openPMD::RecordComponent::SCALAR];
                    setDatasetOptions(dataSet, matcher.get(basename + "/mass"));
                    massComponent.resetDataset(dataSet);
                    massComponent.makeConstant(mass);

                    auto unitMap = convertToUnitDimension(getMassOrZero.dimension());
                    massRecord.setUnitDimension(unitMap);
                    massComponent.setUnitSI(::picongpu::UNIT_MASS);
                    massRecord.setAttribute("macroWeighted", int32_t(false));
                    massRecord.setAttribute("weightingPower", float_64(1.0));
                    massRecord.setAttribute("timeOffset", float_64(0.0));
                }

                // charge
                using hasBoundElectrons = typename pmacc::traits::HasIdentifier<FrameType, boundElectrons>::type;
                plugins::output::GetChargeOrZero<FrameType> const getChargeOrZero;
                if(!hasBoundElectrons::value && getChargeOrZero.hasChargeRatio)
                {
                    const float_64 charge(getChargeOrZero());
                    auto& chargeRecord = record["charge"];
                    auto& chargeComponent = chargeRecord[::openPMD::RecordComponent::SCALAR];
                    setDatasetOptions(dataSet, matcher.get(basename + "/charge"));
                    chargeComponent.resetDataset(dataSet);
                    chargeComponent.makeConstant(charge);

                    auto unitMap = convertToUnitDimension(getChargeOrZero.dimension());
                    chargeRecord.setUnitDimension(unitMap);
                    chargeComponent.setUnitSI(::picongpu::UNIT_CHARGE);
                    chargeRecord.setAttribute("macroWeighted", int32_t(false));
                    chargeRecord.setAttribute("weightingPower", float_64(1.0));
                    chargeRecord.setAttribute("timeOffset", float_64(0.0));
                }
            }

            template<typename Space> // has operator[] -> integer type
            HINLINE void operator()(ThreadParams* params, const Space particleOffset)
            {
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) write species: %1%") % T_SpeciesFilter::getName();
                DataConnector& dc = Environment<>::get().DataConnector();
                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                uint64_t mpiSize = gc.getGlobalSize();
                uint64_t mpiRank = gc.getGlobalRank();
                /* load particle without copy particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(ThisSpecies::FrameType::getName(), true);
                const std::string speciesGroup(T_Species::getName());

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Iteration iteration = series.WRITE_ITERATIONS[params->currentStep];
                const std::string basename = series.particlesPath() + speciesGroup;

                // enforce that the filter interface is fulfilled
                particles::filter::IUnary<typename T_SpeciesFilter::Filter> particleFilter{params->currentStep};
                using usedFilters = bmpl::vector<typename GetPositionFilter<simDim>::type>;
                using MyParticleFilter = typename FilterFactory<usedFilters>::FilterType;
                MyParticleFilter filter;
                /* activate filter pipeline if moving window is activated */
                filter.setStatus(MovingWindow::getInstance().isSlidingWindowActive(params->currentStep));
                filter.setWindowPosition(params->localWindowToDomainOffset, params->window.localDimensions.size);

                using RunParameters_T = StrategyRunParameters<
                    decltype(speciesTmp),
                    decltype(filter),
                    decltype(particleFilter),
                    const Space>;

                using AStrategy = Strategy<openPMDFrameType, RunParameters_T>;
                std::unique_ptr<AStrategy> strategy;

                switch(params->strategy)
                {
                case WriteSpeciesStrategy::ADIOS:
                {
                    using type = StrategyADIOS<openPMDFrameType, RunParameters_T>;
                    strategy = std::unique_ptr<AStrategy>(dynamic_cast<AStrategy*>(new type));
                    break;
                }
                case WriteSpeciesStrategy::HDF5:
                {
                    using type = StrategyHDF5<openPMDFrameType, RunParameters_T>;
                    strategy = std::unique_ptr<AStrategy>(dynamic_cast<AStrategy*>(new type));
                    break;
                }
                }


                /* count total number of particles on the device */
                log<picLog::INPUT_OUTPUT>("openPMD:   (begin) count particles: %1%") % T_SpeciesFilter::getName();
                uint64_cu const myNumParticles = pmacc::CountParticles::countOnDevice<CORE + BORDER>(
                    *speciesTmp,
                    *(params->cellDescription),
                    params->localWindowToDomainOffset,
                    params->window.localDimensions.size,
                    particleFilter);
                uint64_t allNumParticles[mpiSize];
                uint64_t globalNumParticles = 0;
                uint64_t myParticleOffset = 0;

                // avoid deadlock between not finished pmacc tasks and mpi blocking
                // collectives
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
                log<picLog::INPUT_OUTPUT>("openPMD:   ( end ) count particles: %1% = %2%") % T_SpeciesFilter::getName()
                    % globalNumParticles;

                ::openPMD::ParticleSpecies& particleSpecies = iteration.particles[speciesGroup];

                // copy over particles to host
                openPMDFrameType hostFrame;

                strategy->malloc(T_SpeciesFilter::getName(), hostFrame, myNumParticles);
                RunParameters_T runParameters(
                    dc,
                    *params,
                    speciesTmp,
                    filter,
                    particleFilter,
                    particleOffset,
                    myNumParticles,
                    globalNumParticles);
                if(globalNumParticles > 0)
                {
                    strategy->prepare(T_SpeciesFilter::getName(), hostFrame, std::move(runParameters));
                }
                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) write particle records for %1%")
                    % T_SpeciesFilter::getName();

                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, openPMD::ParticleAttribute<bmpl::_1>>
                    writeToOpenPMD;
                writeToOpenPMD(
                    params,
                    hostFrame,
                    particleSpecies,
                    basename,
                    myNumParticles,
                    globalNumParticles,
                    myParticleOffset);

                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) free memory: %1%") % T_SpeciesFilter::getName();
                /* free host memory */
                strategy->free(hostFrame);
                log<picLog::INPUT_OUTPUT>("openPMD:  (end) free memory: %1%") % T_SpeciesFilter::getName();

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing species: %1%") % T_SpeciesFilter::getName();

                /* write species counter table to openPMD storage */
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) writing particle patches for %1%")
                    % T_SpeciesFilter::getName();
                {
                    using index_t = uint64_t;
                    ::openPMD::Datatype const datatype = ::openPMD::determineDatatype<index_t>();
                    // not const, we'll switch out the JSON config
                    ::openPMD::Dataset ds(datatype, {mpiSize});

                    ::openPMD::ParticlePatches particlePatches = particleSpecies.particlePatches;
                    ::openPMD::PatchRecordComponent numParticles
                        = particlePatches["numParticles"][::openPMD::RecordComponent::SCALAR];
                    ::openPMD::PatchRecordComponent numParticlesOffset
                        = particlePatches["numParticlesOffset"][::openPMD::RecordComponent::SCALAR];

                    setDatasetOptions(ds, params->jsonMatcher->get(basename + "/particlePatches/numParticles"));
                    numParticles.resetDataset(ds);
                    setDatasetOptions(ds, params->jsonMatcher->get(basename + "/particlePatches/numParticlesOffset"));
                    numParticlesOffset.resetDataset(ds);

                    /* It is safe to use the mpi rank to write the data even if the rank can differ between simulation
                     * runs. During the restart the plugin is using patch information to find the corresponding data.
                     */
                    numParticles.store<index_t>(mpiRank, myNumParticles);
                    numParticlesOffset.store<index_t>(mpiRank, myParticleOffset);

                    ::openPMD::PatchRecord offset = particlePatches["offset"];
                    ::openPMD::PatchRecord extent = particlePatches["extent"];
                    auto const patchExtent = params->window.localDimensions.size;

                    for(size_t d = 0; d < simDim; ++d)
                    {
                        ::openPMD::PatchRecordComponent offset_x = offset[name_lookup[d]];
                        ::openPMD::PatchRecordComponent extent_x = extent[name_lookup[d]];
                        setDatasetOptions(
                            ds,
                            params->jsonMatcher->get(basename + "/particlePatches/offset/" + name_lookup[d]));
                        offset_x.resetDataset(ds);
                        setDatasetOptions(
                            ds,
                            params->jsonMatcher->get(basename + "/particlePatches/extent/" + name_lookup[d]));
                        extent_x.resetDataset(ds);

                        offset_x.store<index_t>(mpiRank, particleOffset[d]);
                        extent_x.store<index_t>(mpiRank, patchExtent[d]);
                    }

                    /* openPMD ED-PIC: additional attributes */
                    setParticleAttributes(
                        particleSpecies,
                        *params->jsonMatcher,
                        series.particlesPath() + speciesGroup);
                    params->openPMDSeries->flush();
                }

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing particle patches for %1%")
                    % T_SpeciesFilter::getName();
            }
        };


    } // namespace openPMD

} // namespace picongpu
