/* Copyright 2014-2024 Rene Widera, Felix Schmitt, Axel Huebl,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/particles/traits/GetShape.hpp"
#    include "picongpu/particles/traits/GetSpeciesFlagName.hpp"
#    include "picongpu/plugins/ISimulationPlugin.hpp"
#    include "picongpu/plugins/common/MPIHelpers.hpp"
#    include "picongpu/plugins/kernel/CopySpecies.kernel"
#    include "picongpu/plugins/openPMD/openPMDDimension.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/plugins/openPMD/writer/ParticleAttribute.hpp"
#    include "picongpu/plugins/openPMD/writer/misc.hpp"
#    include "picongpu/plugins/output/ConstSpeciesAttributes.hpp"
#    include "picongpu/plugins/output/WriteSpeciesCommon.hpp"

#    include <pmacc/assert.hpp>
#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/eventSystem/events/kernelEvents.hpp>
#    include <pmacc/mappings/kernel/AreaMapping.hpp>
#    include <pmacc/mappings/kernel/RangeMapping.hpp>
#    include <pmacc/memory/buffers/GridBuffer.hpp>
#    include <pmacc/meta/conversion/MakeSeq.hpp>
#    include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#    include <pmacc/particles/ParticleDescription.hpp>
#    include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#    include <pmacc/particles/memory/frames/Frame.hpp>
#    include <pmacc/particles/operations/ConcatListOfFrames.hpp>
#    include <pmacc/particles/operations/CountParticles.hpp>
#    include <pmacc/particles/particleFilter/FilterFactory.hpp>
#    include <pmacc/particles/particleFilter/PositionFilter.hpp>

#    include <boost/mpl/placeholders.hpp>

#    include <algorithm>
#    include <exception>
#    include <type_traits> // std::remove_reference_t
#    include <vector>

namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        template<typename SpeciesTmp, typename Filter, typename ParticleFilter, typename T_ParticleOffset>
        struct StrategyRunParameters
        {
            using SpeciesType = SpeciesTmp;
            pmacc::DataConnector& dc;
            ThreadParams& params;
            SpeciesTmp& speciesTmp;
            Filter& filter;
            ParticleFilter& particleFilter;
            T_ParticleOffset& particleToTotalDomainOffset;
            uint64_t globalNumParticles;

            StrategyRunParameters(
                pmacc::DataConnector& c_dc,
                ThreadParams& c_params,
                SpeciesTmp& c_speciesTmp,
                Filter& c_filter,
                ParticleFilter& c_particleFilter,
                T_ParticleOffset& c_particleToTotalDomainOffset,
                uint64_t c_globalNumParticles)
                : dc(c_dc)
                , params(c_params)
                , speciesTmp(c_speciesTmp)
                , filter(c_filter)
                , particleFilter(c_particleFilter)
                , particleToTotalDomainOffset(c_particleToTotalDomainOffset)
                , globalNumParticles(c_globalNumParticles)
            {
            }
        };

        template<typename T_ParticleDescription, typename T_RunParameters>
        struct Strategy
        {
            using FrameType = Frame<OperatorCreateVectorBox, T_ParticleDescription>;
            using BufferType = Frame<OperatorCreateAlpakaBuffer, T_ParticleDescription>;

            using RunParameters = T_RunParameters;

            BufferType buffers;
            FrameType frame;

            virtual FrameType malloc(std::string name, uint64_cu const myNumParticles) = 0;

            virtual void prepare(
                uint32_t const currentStep,
                std::string name,
                RunParameters,
                std::optional<ChunkDescription> const& chunkDesc,
                bool verbose_log = true)
                = 0;

            virtual ~Strategy() = default;
        };

        /*
         * Use double buffer.
         */
        template<typename T_ParticleDescription, typename RunParameters>
        struct StrategyADIOS : Strategy<T_ParticleDescription, RunParameters>
        {
            using FrameType = typename Strategy<T_ParticleDescription, RunParameters>::FrameType;
            uint64_cu m_myNumParticles = 0;

            FrameType malloc(std::string name, uint64_cu const myNumParticles) override
            {
                /* malloc host memory */
                m_myNumParticles = myNumParticles;
                log<picLog::INPUT_OUTPUT>("openPMD:   (begin) malloc host memory: %1%") % name;
                meta::ForEach<typename T_ParticleDescription::ValueTypeSeq, MallocHostMemory<boost::mpl::_1>>
                    mallocMem;
                mallocMem(this->buffers, this->frame, myNumParticles);
                log<picLog::INPUT_OUTPUT>("openPMD:   ( end ) malloc host memory: %1%") % name;
                return this->frame;
            }

            void prepare(
                uint32_t const currentStep,
                std::string name,
                RunParameters rp,
                std::optional<ChunkDescription> const& chunkDesc,
                bool verbose_log) override
            {
                if(verbose_log)
                {
                    log<picLog::INPUT_OUTPUT>("openPMD:   (begin) copy particle host (with hierarchy) to "
                                              "host (without hierarchy): %1%")
                        % name;
                }

                int particlesProcessed = 0;
                auto const areaMapper = makeAreaMapper<CORE + BORDER>(*(rp.params.cellDescription));
                auto rangeMapper
                    = chunkDesc.has_value()
                          ? makeRangeMapper(areaMapper, chunkDesc->beginSupercellIdx, chunkDesc->endSupercellIdx)
                          : makeRangeMapper(areaMapper);

                pmacc::particles::operations::ConcatListOfFrames concatListOfFrames{};

#    if (ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)
                auto mallocMCBuffer
                    = rp.dc.template get<MallocMCBuffer<DeviceHeap>>(MallocMCBuffer<DeviceHeap>::getName());
                auto particlesBox = rp.speciesTmp->getHostParticlesBox(mallocMCBuffer->getOffset());
#    else
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
                eventSystem::startOperation(ITask::TASK_HOST);

#    endif
                concatListOfFrames(
                    particlesProcessed,
                    this->frame,
                    particlesBox,
                    rp.filter,
                    rp.particleToTotalDomainOffset,
                    totalCellIdx_,
                    rangeMapper,
                    rp.particleFilter);

                /* this costs a little bit of time but writing to external is
                 * slower in general */
                PMACC_VERIFY(
                    (uint64_cu) particlesProcessed == chunkDesc.has_value() ? chunkDesc->numberOfParticles
                                                                            : m_myNumParticles);
            }
        };

        /*
         * Use mapped memory.
         */
        template<typename T_ParticleDescription, typename RunParameters>
        struct StrategyHDF5 : Strategy<T_ParticleDescription, RunParameters>
        {
            using FrameType = typename Strategy<T_ParticleDescription, RunParameters>::FrameType;
            uint64_cu m_myNumParticles = 0;

            FrameType malloc(std::string name, uint64_cu const myNumParticles) override
            {
                log<picLog::INPUT_OUTPUT>("openPMD:  (begin) malloc mapped memory: %1%") % name;
                m_myNumParticles = myNumParticles;
                /*malloc mapped memory*/
                meta::ForEach<typename T_ParticleDescription::ValueTypeSeq, MallocMappedMemory<boost::mpl::_1>>
                    mallocMem;
                mallocMem(this->buffers, this->frame, myNumParticles);
                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) malloc mapped memory: %1%") % name;
                return this->frame;
            }

            void prepare(
                uint32_t currentStep,
                std::string name,
                RunParameters rp,
                std::optional<ChunkDescription> const& chunkDesc,
                bool verbose_log) override
            {
                if(verbose_log)
                {
                    log<picLog::INPUT_OUTPUT>("openPMD:  (begin) copy particle to host: %1%") % name;
                }

                GridBuffer<int, DIM1> counterBuffer(DataSpace<DIM1>(1));
                auto const areaMapper = makeAreaMapper<CORE + BORDER>(*(rp.params.cellDescription));
                auto rangeMapper
                    = chunkDesc.has_value()
                          ? makeRangeMapper(areaMapper, chunkDesc->beginSupercellIdx, chunkDesc->endSupercellIdx)
                          : makeRangeMapper(areaMapper);

                /* this sanity check costs a little bit of time but hdf5 writing is
                 * slower */
                PMACC_LOCKSTEP_KERNEL(CopySpecies{})
                    .config(rangeMapper.getGridDim(), *rp.speciesTmp)(
                        counterBuffer.getDeviceBuffer().data(),
                        this->frame,
                        rp.speciesTmp->getDeviceParticlesBox(),
                        rp.filter,
                        rp.particleToTotalDomainOffset,
                        totalCellIdx_,
                        rangeMapper,
                        rp.particleFilter);
                counterBuffer.deviceToHost();
                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) copy particle to host: %1%") % name;
                eventSystem::getTransactionEvent().waitForFinished();
                log<picLog::INPUT_OUTPUT>("openPMD:  all events are finished: %1%") % name;

                PMACC_VERIFY(
                    (uint64_t) counterBuffer.getHostBuffer().getDataBox()[0] == chunkDesc.has_value()
                        ? chunkDesc->numberOfParticles
                        : m_myNumParticles);
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

            using usedFilters = pmacc::mp_list<typename GetPositionFilter<simDim>::type>;
            using MyParticleFilter = typename FilterFactory<usedFilters>::FilterType;

            /* delete multiMask and localCellIdx in openPMD particle*/
            using TypesToDelete = pmacc::mp_list<multiMask, localCellIdx>;
            using ParticleCleanedAttributeList = typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type;

            /* add totalCellIdx for openPMD particle*/
            using ParticleNewAttributeList = MakeSeq_t<ParticleCleanedAttributeList, totalCellIdx>;

            using NewParticleDescription =
                typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type;

            static void setParticleAttributes(
                ::openPMD::ParticleSpecies& record,
                uint64_t const globalNumParticles,
                AbstractJsonMatcher& matcher,
                std::string const& basename)
            {
                float_64 const particleShape(picongpu::traits::GetShape<ThisSpecies>::type::assignmentFunctionOrder);
                record.setAttribute("particleShape", particleShape);

                traits::GetSpeciesFlagName<ThisSpecies, current<>> currentDepositionName;
                std::string const currentDeposition(currentDepositionName());
                record.setAttribute("currentDeposition", currentDeposition.c_str());

                traits::GetSpeciesFlagName<ThisSpecies, particlePusher<>> particlePushName;
                std::string const particlePush(particlePushName());
                record.setAttribute("particlePush", particlePush.c_str());

                traits::GetSpeciesFlagName<ThisSpecies, interpolation<>> particleInterpolationName;
                std::string const particleInterpolation(particleInterpolationName());
                record.setAttribute("particleInterpolation", particleInterpolation.c_str());

                std::string const particleSmoothing("none");
                record.setAttribute("particleSmoothing", particleSmoothing.c_str());

                // now we have a map in a writeable format with all zeroes
                // for each record copy it and modify the copy, e.g.

                // const records stuff
                ::openPMD::Datatype dataType = ::openPMD::Datatype::DOUBLE;
                ::openPMD::Extent extent = {globalNumParticles};
                ::openPMD::Dataset dataSet = ::openPMD::Dataset(dataType, extent);

                // mass
                plugins::output::GetMassOrZero<FrameType> const getMassOrZero;
                if(getMassOrZero.hasMassRatio)
                {
                    float_64 const mass(getMassOrZero());
                    auto& massRecord = record["mass"];
                    auto& massComponent = massRecord[::openPMD::RecordComponent::SCALAR];
                    dataSet.options = matcher.get(basename + "/mass");
                    massComponent.resetDataset(dataSet);
                    massComponent.makeConstant(mass);

                    auto unitMap = convertToUnitDimension(getMassOrZero.dimension());
                    massRecord.setUnitDimension(unitMap);
                    massComponent.setUnitSI(::picongpu::sim.unit.mass());
                    massRecord.setAttribute("macroWeighted", int32_t(false));
                    massRecord.setAttribute("weightingPower", float_64(1.0));
                    massRecord.setAttribute("timeOffset", float_64(0.0));
                }

                // charge
                using hasBoundElectrons = typename pmacc::traits::HasIdentifier<FrameType, boundElectrons>::type;
                plugins::output::GetChargeOrZero<FrameType> const getChargeOrZero;
                if(!hasBoundElectrons::value && getChargeOrZero.hasChargeRatio)
                {
                    float_64 const charge(getChargeOrZero());
                    auto& chargeRecord = record["charge"];
                    auto& chargeComponent = chargeRecord[::openPMD::RecordComponent::SCALAR];
                    dataSet.options = matcher.get(basename + "/charge");
                    chargeComponent.resetDataset(dataSet);
                    chargeComponent.makeConstant(charge);

                    auto unitMap = convertToUnitDimension(getChargeOrZero.dimension());
                    chargeRecord.setUnitDimension(unitMap);
                    chargeComponent.setUnitSI(::picongpu::sim.unit.charge());
                    chargeRecord.setAttribute("macroWeighted", int32_t(false));
                    chargeRecord.setAttribute("weightingPower", float_64(1.0));
                    chargeRecord.setAttribute("timeOffset", float_64(0.0));
                }
            }

            static void writeParticlePatches(
                ThreadParams* params,
                DataSpace<simDim> const& particleToTotalDomainOffset,
                ::openPMD::Series& series,
                ::openPMD::ParticleSpecies& particleSpecies,
                std::string const& basename,
                std::string const& speciesGroup,
                uint64_t mpiSize,
                uint64_t mpiRank,
                size_t myParticleOffset,
                size_t myNumParticles,
                size_t globalNumParticles)
            {
                /* write species counter table to openPMD storage */
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) writing particle patches for %1%")
                    % T_SpeciesFilter::getName();
                using index_t = uint64_t;
                ::openPMD::Datatype const datatype = ::openPMD::determineDatatype<index_t>();
                // not const, we'll switch out the JSON config
                ::openPMD::Dataset ds(datatype, {mpiSize});

                ::openPMD::ParticlePatches particlePatches = particleSpecies.particlePatches;
                ::openPMD::PatchRecordComponent numParticles
                    = particlePatches["numParticles"][::openPMD::RecordComponent::SCALAR];
                ::openPMD::PatchRecordComponent numParticlesOffset
                    = particlePatches["numParticlesOffset"][::openPMD::RecordComponent::SCALAR];

                ds.options = params->jsonMatcher->get(basename + "/particlePatches/numParticles");
                numParticles.resetDataset(ds);
                ds.options = params->jsonMatcher->get(basename + "/particlePatches/numParticlesOffset");
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
                    ds.options = params->jsonMatcher->get(basename + "/particlePatches/offset/" + name_lookup[d]);
                    offset_x.resetDataset(ds);
                    ds.options = params->jsonMatcher->get(basename + "/particlePatches/extent/" + name_lookup[d]);
                    extent_x.resetDataset(ds);

                    auto const totalPatchOffset
                        = particleToTotalDomainOffset[d] + params->localWindowToDomainOffset[d];
                    offset_x.store<index_t>(mpiRank, totalPatchOffset);
                    extent_x.store<index_t>(mpiRank, patchExtent[d]);
                }

                /* openPMD ED-PIC: additional attributes */
                setParticleAttributes(
                    particleSpecies,
                    globalNumParticles,
                    *params->jsonMatcher,
                    series.particlesPath() + speciesGroup);
                params->m_dumpTimes.now<std::chrono::milliseconds>("\tFlush species " + T_SpeciesFilter::getName());
                particleSpecies.seriesFlush(PreferredFlushTarget::Buffer);
                params->m_dumpTimes.now<std::chrono::milliseconds>("\tFinished flush species");

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing particle patches for %1%")
                    % T_SpeciesFilter::getName();
            }

            template<typename AStrategy>
            static void writeParticlesWithoutAnySubdividing(
                ThreadParams* params,
                uint32_t const currentStep,
                ::openPMD::Series& series,
                ::openPMD::ParticleSpecies& particleSpecies,
                std::string const& basename,
                std::string const& speciesGroup,
                std::unique_ptr<AStrategy> strategy,
                DataConnector& dc,
                typename AStrategy::RunParameters::SpeciesType& speciesTmp,
                particles::filter::IUnary<typename T_SpeciesFilter::Filter>& particleFilter,
                DataSpace<simDim> const& particleToTotalDomainOffset)
            {
                /* count total number of particles on the device */
                log<picLog::INPUT_OUTPUT>("openPMD:   (begin) count particles: %1%") % T_SpeciesFilter::getName();
                uint64_cu const myNumParticles = pmacc::CountParticles::countOnDevice<CORE + BORDER>(
                    *speciesTmp,
                    *(params->cellDescription),
                    params->localWindowToDomainOffset,
                    params->window.localDimensions.size,
                    particleFilter);

                uint64_t globalNumParticles = 0;
                uint64_t myParticleOffset = 0;

                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                uint64_t mpiSize = gc.getGlobalSize();
                uint64_t mpiRank = gc.getGlobalRank();
                std::vector<uint64_t> allNumParticles(mpiSize);


                MyParticleFilter filter;
                filter.setWindowPosition(params->localWindowToDomainOffset, params->window.localDimensions.size);

                // avoid deadlock between not finished pmacc tasks and mpi blocking
                // collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    &myNumParticles,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    allNumParticles.data(),
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    gc.getCommunicator().getMPIComm()));

                for(uint64_t i = 0; i < mpiSize; ++i)
                {
                    globalNumParticles += allNumParticles[i];
                    if(i < mpiRank)
                        myParticleOffset += allNumParticles[i];
                }

                log<picLog::INPUT_OUTPUT>("openPMD: species '%1%': global particle count=%2%")
                    % T_SpeciesFilter::getName() % globalNumParticles;

                auto hostFrame = strategy->malloc(T_SpeciesFilter::getName(), myNumParticles);

                {
                    meta::ForEach<
                        typename NewParticleDescription::ValueTypeSeq,
                        openPMD::InitParticleAttribute<boost::mpl::_1>>
                        initParticleAttributes;
                    initParticleAttributes(params, particleSpecies, basename, globalNumParticles);
                }


                using RunParameters = typename AStrategy::RunParameters;
                RunParameters runParameters(
                    dc,
                    *params,
                    speciesTmp,
                    filter,
                    particleFilter,
                    particleToTotalDomainOffset,
                    globalNumParticles);

                if(myNumParticles > 0)
                {
                    strategy->prepare(currentStep, T_SpeciesFilter::getName(), runParameters, std::nullopt);
                }
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) write particle records for %1%")
                    % T_SpeciesFilter::getName();

                std::stringstream description;
                description << "\tprepare";
                params->m_dumpTimes.now<std::chrono::milliseconds>(description.str());

                size_t writtenBytes = 0;

                meta::
                    ForEach<typename NewParticleDescription::ValueTypeSeq, openPMD::ParticleAttribute<boost::mpl::_1>>
                        writeToOpenPMD;
                writeToOpenPMD(
                    params,
                    hostFrame,
                    particleSpecies,
                    basename,
                    myNumParticles,
                    globalNumParticles,
                    myParticleOffset,
                    writtenBytes,
                    /* verbose_log = */ true);

                description = std::stringstream();
                description << ": " << writtenBytes << " bytes for " << myNumParticles << " particles from offset "
                            << myParticleOffset;
                params->m_dumpTimes.append(description.str());

                log<picLog::INPUT_OUTPUT>("openPMD: flush particle records for %1%") % T_SpeciesFilter::getName();

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                params->m_dumpTimes.now<std::chrono::milliseconds>("\tflush");
                particleSpecies.seriesFlush(PreferredFlushTarget::Disk);
                params->m_dumpTimes.now<std::chrono::milliseconds>("\tend");


                log<picLog::INPUT_OUTPUT>("openPMD: (end) write particle records for %1%")
                    % T_SpeciesFilter::getName();


                strategy.reset();

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing species: %1%") % T_SpeciesFilter::getName();

                writeParticlePatches(
                    params,
                    particleToTotalDomainOffset,
                    series,
                    particleSpecies,
                    basename,
                    speciesGroup,
                    mpiSize,
                    mpiRank,
                    myParticleOffset,
                    myNumParticles,
                    globalNumParticles);
            }

            template<typename AStrategy>
            static void writeParticlesByChunks(
                ThreadParams* params,
                uint32_t const currentStep,
                ::openPMD::Series& series,
                ::openPMD::ParticleSpecies& particleSpecies,
                std::string const& basename,
                std::string const& speciesGroup,
                std::unique_ptr<AStrategy> strategy,
                DataConnector& dc,
                typename AStrategy::RunParameters::SpeciesType& speciesTmp,
                particles::filter::IUnary<typename T_SpeciesFilter::Filter>& particleFilter,
                DataSpace<simDim> const& particleToTotalDomainOffset)
            {
                uint64_t myNumParticles = 0;
                uint64_t globalNumParticles = 0;
                uint64_t myParticleOffset = 0;

                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                uint64_t mpiSize = gc.getGlobalSize();
                uint64_t mpiRank = gc.getGlobalRank();

                MyParticleFilter filter;
                filter.setWindowPosition(params->localWindowToDomainOffset, params->window.localDimensions.size);

                constexpr size_t particleSizeInByte = AStrategy::FrameType::ParticleType::sizeInByte();
                ParticleIoChunkInfo particleIoChunkInfo
                    = createSupercellRangeChunks(params, speciesTmp, particleFilter, particleSizeInByte);
                uint64_t const requiredDumpRounds = particleIoChunkInfo.ranges.size();

                /* count total number of particles on the device */
                log<picLog::INPUT_OUTPUT>(
                    "openPMD: species '%1%': particles=%2% in %3% chunks, largest data chunk size %4% byte")
                    % T_SpeciesFilter::getName() % particleIoChunkInfo.totalNumParticles % requiredDumpRounds
                    % (particleIoChunkInfo.largestChunk * particleSizeInByte);

                myNumParticles = particleIoChunkInfo.totalNumParticles;
                std::vector<uint64_t> allNumParticles(mpiSize);

                // avoid deadlock between not finished pmacc tasks and mpi blocking
                // collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    &myNumParticles,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    allNumParticles.data(),
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    gc.getCommunicator().getMPIComm()));

                for(uint64_t i = 0; i < mpiSize; ++i)
                {
                    globalNumParticles += allNumParticles[i];
                    if(i < mpiRank)
                        myParticleOffset += allNumParticles[i];
                }

                // @todo combine this and the MPI_Gather above to a single gather for better scaling
                std::vector<uint64_t> numRounds(mpiSize);

                MPI_CHECK(MPI_Allgather(
                    &requiredDumpRounds,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    numRounds.data(),
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    gc.getCommunicator().getMPIComm()));

                uint64_t globalNumDumpRounds = requiredDumpRounds;
                for(uint64_t i = 0; i < mpiSize; ++i)
                    globalNumDumpRounds = std::max(globalNumDumpRounds, numRounds[i]);


                log<picLog::INPUT_OUTPUT>(
                    "openPMD: species '%1%': global particle count=%2%, number of dumping iterations=%3%")
                    % T_SpeciesFilter::getName() % globalNumParticles % globalNumDumpRounds;

                auto hostFrame = strategy->malloc(T_SpeciesFilter::getName(), particleIoChunkInfo.largestChunk);

                {
                    meta::ForEach<
                        typename NewParticleDescription::ValueTypeSeq,
                        openPMD::InitParticleAttribute<boost::mpl::_1>>
                        initParticleAttributes;
                    initParticleAttributes(params, particleSpecies, basename, globalNumParticles);
                }

                /** Offset within our global chunk where we are allowed to write particles too.
                 *  The offset is updated each dumping iteration.
                 */
                size_t particleOffset = 0u;

                uint64_t dumpIteration = 0u;
                // To write all metadata for particles we need to perform dumping once even if we have no
                // particles
                do
                {
                    ChunkDescription chunk;
                    if(dumpIteration < particleIoChunkInfo.ranges.size())
                    {
                        chunk = particleIoChunkInfo.ranges[dumpIteration];
                    }

                    using RunParameters = typename AStrategy::RunParameters;
                    RunParameters runParameters(
                        dc,
                        *params,
                        speciesTmp,
                        filter,
                        particleFilter,
                        particleToTotalDomainOffset,
                        globalNumParticles);

                    if(chunk.numberOfParticles > 0)
                    {
                        strategy->prepare(currentStep, T_SpeciesFilter::getName(), runParameters, chunk);
                    }
                    log<picLog::INPUT_OUTPUT>("openPMD: (begin) write particle records for %1%, dumping round %2%")
                        % T_SpeciesFilter::getName() % dumpIteration;

                    std::stringstream description;
                    description << "\tslice " << dumpIteration << " prepare";
                    params->m_dumpTimes.now<std::chrono::milliseconds>(description.str());

                    size_t writtenBytes = 0;

                    meta::ForEach<
                        typename NewParticleDescription::ValueTypeSeq,
                        openPMD::ParticleAttribute<boost::mpl::_1>>
                        writeToOpenPMD;
                    writeToOpenPMD(
                        params,
                        hostFrame,
                        particleSpecies,
                        basename,
                        chunk.numberOfParticles,
                        globalNumParticles,
                        myParticleOffset + particleOffset,
                        writtenBytes,
                        /* verbose_log = */ true);

                    description = std::stringstream();
                    description << ": " << writtenBytes << " bytes for " << chunk.numberOfParticles
                                << " particles from offset " << (myParticleOffset + particleOffset);
                    params->m_dumpTimes.append(description.str());

                    log<picLog::INPUT_OUTPUT>("openPMD: flush particle records for %1%, dumping round %2%")
                        % T_SpeciesFilter::getName() % dumpIteration;

                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    eventSystem::getTransactionEvent().waitForFinished();
                    params->m_dumpTimes.now<std::chrono::milliseconds>(
                        "\tslice " + std::to_string(dumpIteration) + " flush");
                    particleSpecies.seriesFlush(PreferredFlushTarget::Disk);
                    params->m_dumpTimes.now<std::chrono::milliseconds>(
                        "\tslice " + std::to_string(dumpIteration) + " end");


                    log<picLog::INPUT_OUTPUT>("openPMD: (end) write particle records for %1%, dumping round %2%")
                        % T_SpeciesFilter::getName() % dumpIteration;

                    particleOffset += chunk.numberOfParticles;
                    ++dumpIteration;
                } while(dumpIteration < globalNumDumpRounds);

                strategy.reset();

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing species: %1%") % T_SpeciesFilter::getName();

                writeParticlePatches(
                    params,
                    particleToTotalDomainOffset,
                    series,
                    particleSpecies,
                    basename,
                    speciesGroup,
                    mpiSize,
                    mpiRank,
                    myParticleOffset,
                    myNumParticles,
                    globalNumParticles);
            }

            static void writeParticlePatchesBySupercells(
                ThreadParams* params,
                DataSpace<simDim> const& particleToTotalDomainOffset,
                ParticleIoChunkInfo particleIoChunkInfo,
                ::openPMD::Series& series,
                ::openPMD::ParticleSpecies& particleSpecies,
                std::string const& basename,
                std::string const& speciesGroup,
                GridController<simDim>& gc,
                uint64_t mpiSize,
                uint64_t mpiRank,
                size_t myParticleOffset,
                size_t myNumParticles,
                size_t globalNumParticles)
            {
                /* write species counter table to openPMD storage */
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) writing particle patches for %1%")
                    % T_SpeciesFilter::getName();

                size_t numPatches = particleIoChunkInfo.ranges.size();
                { // Verify that each rank contributes the same number of patches
                    size_t maximum, minimum;
                    MPI_Allreduce(
                        &numPatches,
                        &minimum,
                        1,
                        MPI_Types<size_t>().value,
                        MPI_MIN,
                        gc.getCommunicator().getMPIComm());
                    MPI_Allreduce(
                        &numPatches,
                        &maximum,
                        1,
                        MPI_Types<size_t>().value,
                        MPI_MAX,
                        gc.getCommunicator().getMPIComm());

                    if(minimum != maximum)
                    {
                        throw std::runtime_error(
                            "Ranks have different number of supercells: Minimum number = " + std::to_string(minimum)
                            + ", maximum number: " + std::to_string(maximum) + ". Will abort.");
                    }
                }

                auto const areaMapper = makeAreaMapper<CORE + BORDER>(*params->cellDescription);
                auto const baseGridDim = areaMapper.getGridDim();
                auto const rangeMapper = makeRangeMapper(areaMapper);

                using index_t = uint64_t;
                std::vector<index_t> bufferNumParticlesOffset(numPatches);
                std::vector<index_t> bufferNumParticles(numPatches);
                std::array<std::vector<index_t>, simDim> bufferOffset;
                std::array<std::vector<index_t>, simDim> bufferExtent;
                for(size_t d = 0; d < simDim; ++d)
                {
                    bufferOffset[d].resize(numPatches);
                    bufferExtent[d].resize(numPatches);
                }

                ::openPMD::Datatype const datatype = ::openPMD::determineDatatype<index_t>();
                // not const, we'll switch out the JSON config
                ::openPMD::Dataset ds(datatype, {numPatches * mpiSize});

                ::openPMD::ParticlePatches particlePatches = particleSpecies.particlePatches;
                ::openPMD::PatchRecordComponent numParticles
                    = particlePatches["numParticles"][::openPMD::RecordComponent::SCALAR];
                ::openPMD::PatchRecordComponent numParticlesOffset
                    = particlePatches["numParticlesOffset"][::openPMD::RecordComponent::SCALAR];

                auto const totalPatchOffset = particleToTotalDomainOffset + params->localWindowToDomainOffset;
                auto const totalPatchExtent = params->window.localDimensions.size;
                std::array<double, simDim> singlePatchExtent;
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    singlePatchExtent[d] = (double) totalPatchExtent[d] / baseGridDim[d];
                }

                size_t runningOffset = myParticleOffset;
                for(size_t i = 0; i < numPatches; ++i)
                {
                    auto const& supercell = particleIoChunkInfo.ranges.at(i);
                    if(supercell.endSupercellIdx != supercell.beginSupercellIdx + 1)
                    {
                        throw std::runtime_error(
                            "Grouping of supercells for load balancing output currently not implemented.");
                    }
                    bufferNumParticlesOffset[i] = runningOffset;
                    bufferNumParticles[i] = supercell.numberOfParticles;
                    runningOffset += supercell.numberOfParticles;
                    DataSpace<simDim> const superCellIdxND
                        = rangeMapper.getSuperCellIndex(supercell.beginSupercellIdx);
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        bufferExtent[d][i] = singlePatchExtent[d];
                        // No idea why the -1 is needed, but the results do not fit otherwise
                        // Something weird going on in supercell indexing
                        bufferOffset[d][i]
                            = totalPatchOffset[d] + index_t(singlePatchExtent[d] * (superCellIdxND[d] - 1));
                    }
                }

                ds.options = params->jsonMatcher->get(basename + "/particlePatches/numParticles");
                numParticles.resetDataset(ds);
                ds.options = params->jsonMatcher->get(basename + "/particlePatches/numParticlesOffset");
                numParticlesOffset.resetDataset(ds);

                /* It is safe to use the mpi rank to write the data even if the rank can differ between simulation
                 * runs. During the restart the plugin is using patch information to find the corresponding data.
                 */
                numParticles.storeChunk(bufferNumParticles, {mpiRank * numPatches}, {numPatches});
                numParticlesOffset.storeChunk(bufferNumParticlesOffset, {mpiRank * numPatches}, {numPatches});

                ::openPMD::PatchRecord offset = particlePatches["offset"];
                ::openPMD::PatchRecord extent = particlePatches["extent"];

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    ::openPMD::PatchRecordComponent offset_x = offset[name_lookup[d]];
                    ::openPMD::PatchRecordComponent extent_x = extent[name_lookup[d]];
                    ds.options = params->jsonMatcher->get(basename + "/particlePatches/offset/" + name_lookup[d]);
                    offset_x.resetDataset(ds);
                    ds.options = params->jsonMatcher->get(basename + "/particlePatches/extent/" + name_lookup[d]);
                    extent_x.resetDataset(ds);

                    offset_x.storeChunk(bufferOffset[d], {mpiRank * numPatches}, {numPatches});
                    // TODO: Maybe use a constant record for this
                    extent_x.storeChunk(bufferExtent[d], {mpiRank * numPatches}, {numPatches});
                }

                /* openPMD ED-PIC: additional attributes */
                setParticleAttributes(
                    particleSpecies,
                    globalNumParticles,
                    *params->jsonMatcher,
                    series.particlesPath() + speciesGroup);
                params->m_dumpTimes.now<std::chrono::milliseconds>("\tFlush species " + T_SpeciesFilter::getName());
                particleSpecies.seriesFlush(PreferredFlushTarget::Buffer);
                params->m_dumpTimes.now<std::chrono::milliseconds>("\tFinished flush species");

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing particle patches for %1%")
                    % T_SpeciesFilter::getName();
            }

            template<typename AStrategy>
            static void writeParticlesBySupercells(
                ThreadParams* params,
                uint32_t const currentStep,
                ::openPMD::Series& series,
                ::openPMD::ParticleSpecies& particleSpecies,
                std::string const& basename,
                std::string const& speciesGroup,
                std::unique_ptr<AStrategy> strategy,
                DataConnector& dc,
                typename AStrategy::RunParameters::SpeciesType& speciesTmp,
                particles::filter::IUnary<typename T_SpeciesFilter::Filter>& particleFilter,
                DataSpace<simDim> const& particleToTotalDomainOffset)
            {
                uint64_t myNumParticles = 0;
                uint64_t globalNumParticles = 0;
                uint64_t myParticleOffset = 0;

                GridController<simDim>& gc = Environment<simDim>::get().GridController();
                uint64_t mpiSize = gc.getGlobalSize();
                uint64_t mpiRank = gc.getGlobalRank();

                MyParticleFilter filter;
                filter.setWindowPosition(params->localWindowToDomainOffset, params->window.localDimensions.size);

                ParticleIoChunkInfo particleIoChunkInfo
                    = createAChunkForEverySupercell(params, speciesTmp, particleFilter);
                uint64_t const requiredDumpRounds = particleIoChunkInfo.ranges.size();

                /* count total number of particles on the device */
                constexpr size_t particleSizeInByte = AStrategy::FrameType::ParticleType::sizeInByte();
                log<picLog::INPUT_OUTPUT>(
                    "openPMD: species '%1%': particles=%2% in %3% supercells, largest supercell size %4% byte")
                    % T_SpeciesFilter::getName() % particleIoChunkInfo.totalNumParticles % requiredDumpRounds
                    % (particleIoChunkInfo.largestChunk * particleSizeInByte);

                myNumParticles = particleIoChunkInfo.totalNumParticles;
                std::vector<uint64_t> allNumParticles(mpiSize);

                // avoid deadlock between not finished pmacc tasks and mpi blocking
                // collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    &myNumParticles,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    allNumParticles.data(),
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    gc.getCommunicator().getMPIComm()));

                for(uint64_t i = 0; i < mpiSize; ++i)
                {
                    globalNumParticles += allNumParticles[i];
                    if(i < mpiRank)
                        myParticleOffset += allNumParticles[i];
                }

                // @todo combine this and the MPI_Gather above to a single gather for better scaling
                std::vector<uint64_t> numRounds(mpiSize);

                MPI_CHECK(MPI_Allgather(
                    &requiredDumpRounds,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    numRounds.data(),
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    gc.getCommunicator().getMPIComm()));

                uint64_t globalNumDumpRounds = requiredDumpRounds;
                for(uint64_t i = 0; i < mpiSize; ++i)
                    globalNumDumpRounds = std::max(globalNumDumpRounds, numRounds[i]);


                log<picLog::INPUT_OUTPUT>(
                    "openPMD: species '%1%': global particle count=%2%, number of dumping iterations=%3%")
                    % T_SpeciesFilter::getName() % globalNumParticles % globalNumDumpRounds;

                auto hostFrame = strategy->malloc(T_SpeciesFilter::getName(), particleIoChunkInfo.largestChunk);

                {
                    meta::ForEach<
                        typename NewParticleDescription::ValueTypeSeq,
                        openPMD::InitParticleAttribute<boost::mpl::_1>>
                        initParticleAttributes;
                    initParticleAttributes(params, particleSpecies, basename, globalNumParticles);
                }

                /** Offset within our global chunk where we are allowed to write particles too.
                 *  The offset is updated each dumping iteration.
                 */
                size_t particleOffset = 0u;

                uint64_t dumpIteration = 0u;
                // To write all metadata for particles we need to perform dumping once even if we have no
                // particles

                constexpr size_t log_granularity = 100;
                do
                {
                    if(dumpIteration % log_granularity == 0)
                    {
                        log<picLog::INPUT_OUTPUT>("openPMD: (begin) write particle records for %1%, dumping round %2%")
                            % T_SpeciesFilter::getName() % dumpIteration;
                    }
                    ChunkDescription chunk;
                    if(dumpIteration < particleIoChunkInfo.ranges.size())
                    {
                        chunk = particleIoChunkInfo.ranges[dumpIteration];
                    }

                    using RunParameters = typename AStrategy::RunParameters;
                    RunParameters runParameters(
                        dc,
                        *params,
                        speciesTmp,
                        filter,
                        particleFilter,
                        particleToTotalDomainOffset,
                        globalNumParticles);

                    if(chunk.numberOfParticles > 0)
                    {
                        strategy->prepare(
                            currentStep,
                            T_SpeciesFilter::getName(),
                            runParameters,
                            chunk,
                            /* verbose_log = */ false);
                    }

                    std::stringstream description;
                    description << "\tslice " << dumpIteration << " prepare";
                    params->m_dumpTimes.now<std::chrono::milliseconds>(description.str());

                    size_t writtenBytes = 0;

                    meta::ForEach<
                        typename NewParticleDescription::ValueTypeSeq,
                        openPMD::ParticleAttribute<boost::mpl::_1>>
                        writeToOpenPMD;
                    writeToOpenPMD(
                        params,
                        hostFrame,
                        particleSpecies,
                        basename,
                        chunk.numberOfParticles,
                        globalNumParticles,
                        myParticleOffset + particleOffset,
                        writtenBytes,
                        /* verbose_log = */ false);

                    description = std::stringstream();
                    description << ": " << writtenBytes << " bytes for " << chunk.numberOfParticles
                                << " particles from offset " << (myParticleOffset + particleOffset);
                    params->m_dumpTimes.append(description.str());

                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    eventSystem::getTransactionEvent().waitForFinished();
                    params->m_dumpTimes.now<std::chrono::milliseconds>(
                        "\tslice " + std::to_string(dumpIteration) + " flush");
                    particleSpecies.seriesFlush(PreferredFlushTarget::Disk);
                    params->m_dumpTimes.now<std::chrono::milliseconds>(
                        "\tslice " + std::to_string(dumpIteration) + " end");


                    particleOffset += chunk.numberOfParticles;
                    ++dumpIteration;
                } while(dumpIteration < globalNumDumpRounds);

                strategy.reset();

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) writing species: %1%") % T_SpeciesFilter::getName();

                writeParticlePatchesBySupercells(
                    params,
                    particleToTotalDomainOffset,
                    particleIoChunkInfo,
                    series,
                    particleSpecies,
                    basename,
                    speciesGroup,
                    gc,
                    mpiSize,
                    mpiRank,
                    myParticleOffset,
                    myNumParticles,
                    globalNumParticles);
            }

            HINLINE void operator()(
                ThreadParams* params,
                uint32_t const currentStep,
                DataSpace<simDim> const particleToTotalDomainOffset)
            {
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) write species: %1%") % T_SpeciesFilter::getName();
                params->m_dumpTimes.now<std::chrono::milliseconds>(
                    "Begin write species " + T_SpeciesFilter::getName());

                DataConnector& dc = Environment<>::get().DataConnector();

                /* load particle without copy particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(ThisSpecies::FrameType::getName());
                std::string const speciesGroup(T_Species::getName());

                ::openPMD::Series& series = *params->writeOpenPMDSeries;
                ::openPMD::Iteration iteration = series.writeIterations()[currentStep];
                std::string const basename = series.particlesPath() + speciesGroup;

                auto idProvider = dc.get<IdProvider>("globalId");

                // enforce that the filter interface is fulfilled
                particles::filter::IUnary<typename T_SpeciesFilter::Filter> particleFilter(
                    currentStep,
                    idProvider->getDeviceGenerator());

                ::openPMD::ParticleSpecies& particleSpecies = iteration.particles[speciesGroup];

                using RunParameters_T = StrategyRunParameters<
                    decltype(speciesTmp),
                    MyParticleFilter,
                    decltype(particleFilter),
                    DataSpace<simDim> const>;
                using AStrategy = Strategy<NewParticleDescription, RunParameters_T>;

                std::unique_ptr<AStrategy> strategy;

                switch(params->strategy)
                {
                case WriteSpeciesStrategy::ADIOS:
                    {
                        using type = StrategyADIOS<NewParticleDescription, RunParameters_T>;
                        strategy = std::unique_ptr<AStrategy>(dynamic_cast<AStrategy*>(new type));
                        break;
                    }
                case WriteSpeciesStrategy::HDF5:
                    {
                        using type = StrategyHDF5<NewParticleDescription, RunParameters_T>;
                        strategy = std::unique_ptr<AStrategy>(dynamic_cast<AStrategy*>(new type));
                        break;
                    }
                }

                // writeParticlesByChunks(
                writeParticlesBySupercells(
                    params,
                    currentStep,
                    series,
                    particleSpecies,
                    basename,
                    speciesGroup,
                    std::move(strategy),
                    dc,
                    speciesTmp,
                    particleFilter,
                    particleToTotalDomainOffset);
            }
        };


    } // namespace openPMD
} // namespace picongpu

#endif
