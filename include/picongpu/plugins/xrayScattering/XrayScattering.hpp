/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz,
 *                     Juncheng E, Pawel Ordyna
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
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/common/stringHelpers.hpp"

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/param/xrayScattering.param"
#include "picongpu/plugins/xrayScattering/beam/XrayScatteringBeam.hpp"
#include "picongpu/plugins/xrayScattering/XrayScattering.kernel"
#include "picongpu/plugins/xrayScattering/XrayScatteringWriter.hpp"
#include "picongpu/plugins/xrayScattering/xrayScatteringUtilities.hpp"
#include "picongpu/plugins/xrayScattering/GetScatteringVector.hpp"
#include "picongpu/plugins/xrayScattering/DetermineElectronDensitySolver.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Density.def"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/assert.hpp>

#include <boost/filesystem.hpp>
#include <boost/mpl/bool.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <map>

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            using namespace pmacc;
            using namespace picongpu::SI;
            namespace po = boost::program_options;
            using complex_X = pmacc::math::Complex<float_X>;


            /** xrayScattering plugin
             * This  plugin simulates the SAXS scattering amplitude
             * from the particles number density.
             *
             * @tparam T_ParticlesType Scatterers
             **/
            template<typename T_ParticlesType>
            class XrayScattering : public ISimulationPlugin
            {
            private:
                using SuperCellSize = MappingDesc::SuperCellSize;

                MappingDesc cellDescription;
                uint32_t currentStep;

                //! Probing beam characterization
                std::unique_ptr<beam::XrayScatteringBeam> probingBeam;

                // memory:
                using ComplexBuffer = GridBuffer<complex_X, DIM1>;
                std::unique_ptr<ComplexBuffer> amplitude;
                // Needed as long as opePMD-api doesn't support complex values:
                //! Storage for amplitude real part used when dumping data
                std::vector<float_X> realPart;
                //! Storage for amplitude imaginary part used when dumping data
                std::vector<float_X> imgPart;
                // Used only in the distributed mode:
                //! Storage for receiving amplitude data from another node
                std::vector<complex_X> amplitudeReceive;
                //! Number of scattering vectors on initialy last rank
                uint64_t resOfVectors;
                // Used only in the mirrored mode:
                std::vector<complex_X> amplitudeMaster;

                // Variables for plugin options:
                std::string notifyPeriod;
                std::string speciesName;
                std::string pluginName;
                std::string pluginPrefix;
                std::string fileName;
                std::string fileExtension;
                std::string compressionMethod;
                std::string outputPeriod_s;
                std::string memoryLayout;
                //! Plugin functioning mode
                OutputMemoryLayout outputLayout;
                //! Time steps at which the output is dumped
                using SeqOfTimeSlices = std::vector<pluginSystem::TimeSlice>;
                SeqOfTimeSlices outputPeriod;

                /** Range of scattering vector
                 * The scattering vector here is defined as
                 * 4*pi*sin(theta)/lambda, where 2 * theta is the angle between the
                 * incoming k-vector and the scattered one.
                 * See the definition in this paper https://doi.org/10.1063/1.5008289.
                 **/
                float2_X q_min, q_max, q_step;
                //! Number of scattering vectors
                DataSpace<DIM2> numVectors;

                uint32_t totalSimulationCells;

                // Needed to handle the parallelization over multiple hosts.
                bool isMaster;
                uint32_t mpiRank;
                //! Total number of nodes
                uint32_t countRanks;
                //! Number of Times the distributed output was passed along
                uint32_t accumulatedRotations;
                mpi::MPIReduce reduce;

                //! Output writer
                std::unique_ptr<XrayScatteringWriter<float_X>> dataWriter;


            public:
                //! XrayScattering object initializer.
                XrayScattering()
                    : pluginName("xrayScattering: Calculate the SAXS scattering intensity of a "
                                 "species.")
                    , speciesName(T_ParticlesType::FrameType::getName())
                    , pluginPrefix(speciesName + std::string("_xrayScattering"))
                    ,
                    // this is bodged so it passes the verification at
                    // MappingDescription.hpp:79
                    cellDescription(DataSpace<simDim>(SuperCellSize::toRT()))
                    , isMaster(false)
                    , currentStep(0)
                    , accumulatedRotations(0)
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                //! XrayScattering object destructor.
                ~XrayScattering() override
                {
                }


                //! Adds command line options and their descriptions.
                void pluginRegisterHelp(po::options_description& desc) override
                {
                    desc.add_options()(
                        (pluginPrefix + ".period").c_str(),
                        po::value<std::string>(&notifyPeriod),
                        "enable plugin [for each n-th step]")(
                        (pluginPrefix + ".outputPeriod").c_str(),
                        po::value<std::string>(&outputPeriod_s)->default_value("1"),
                        "dump amplitude [for each n-th step]")(
                        (pluginPrefix + ".qx_max").c_str(),
                        po::value<float_X>(&q_max[0])->default_value(5),
                        "reciprocal space range qx_max (A^-1)")(
                        (pluginPrefix + ".qy_max").c_str(),
                        po::value<float_X>(&q_max[1])->default_value(5),
                        "reciprocal space range qy_max (A^-1)")(
                        (pluginPrefix + ".qx_min").c_str(),
                        po::value<float_X>(&q_min[0])->default_value(-5),
                        "reciprocal space range qx_min (A^-1)")(
                        (pluginPrefix + ".qy_min").c_str(),
                        po::value<float_X>(&q_min[1])->default_value(-5),
                        "reciprocal space range qy_min (A^-1)")(
                        (pluginPrefix + ".n_qx").c_str(),
                        po::value<int>(&numVectors[0])->default_value(100),
                        "number of qx")(
                        (pluginPrefix + ".n_qy").c_str(),
                        po::value<int>(&numVectors[1])->default_value(100),
                        "number of qy")(
                        (pluginPrefix + ".file").c_str(),
                        po::value<std::string>(&fileName)->default_value(pluginName + "Output"),
                        "output file name")(
                        (pluginPrefix + ".ext").c_str(),
                        po::value<std::string>(&fileExtension)->default_value("bp"),
                        "openPMD filename extension (this controls the backend "
                        "picked by the openPMD API)")(
                        (pluginPrefix + ".compression").c_str(),
                        po::value<std::string>(&compressionMethod)->default_value(""),
                        "Backend-specific openPMD compression method, e.g., zlib "
                        "(see `adios_config -m` for help)")(
                        (pluginPrefix + ".memoryLayout").c_str(),
                        po::value<std::string>(&memoryLayout)->default_value("mirror"),
                        "Possible values: 'mirror' and 'distribute'"
                        "Output can be mirrored on all Host+Device pairs or"
                        " uniformly distributed over all nodes. Distribute can be used "
                        "when the output array is to big to store the complete "
                        "computed q-space on one device.");
                }


                //! Get plugin name.
                std::string pluginGetName() const override
                {
                    return pluginName;
                }


                //! Sets Mapping description for the xrayScattering plugin.
                void setMappingDescription(MappingDesc* cellDescriptionLoc) override
                {
                    cellDescription = *cellDescriptionLoc;
                }


                void restart(uint32_t timeStep, const std::string restartDirectory) override
                {
                    log<picLog::INPUT_OUTPUT>("XrayScattering : restart not"
                                              "yet implemented - start with zero values");
                    // TODO: Support for restarting.
                }


                void checkpoint(uint32_t timeStep, const std::string restartDirectory) override
                {
                    log<picLog::INPUT_OUTPUT>("XrayScattering : checkpoint not"
                                              "yet implemented - nothing was saved");

                    // TODO: Support for restarting.
                }


            private:
                //! Prepare the plugin in the simulation initialization phase.
                void pluginLoad() override
                {
                    if(!notifyPeriod.empty())
                    {
                        /* Beam has to be initialized later as the domain sizes.
                         *   The value retrieved by getDomainSize in
                         *   CoordinateTransform.hpp is still set to (0,0,0) when the
                         *   XrayScattering object is initialized.
                         */
                        probingBeam = std::make_unique<beam::XrayScatteringBeam>();
                        // Set the steps at which the xrayScattering amplitude is
                        // calculated.
                        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
                        // Set the memory layout in use.
                        std::map<std::string, OutputMemoryLayout> layoutMap;
                        layoutMap["mirror"] = OutputMemoryLayout::Mirror;
                        layoutMap["distribute"] = OutputMemoryLayout::Distribute;
                        outputLayout = layoutMap.at(memoryLayout);

                        GridController<simDim>& gc = Environment<simDim>::get().GridController();
                        mpiRank = gc.getGlobalRank();
                        isMaster = (mpiRank == 0);

                        // Prepare amplitude buffer:
                        uint32_t bufferSize;
                        auto totalNumVectors = numVectors.productOfComponents();
                        if(outputLayout == OutputMemoryLayout::Mirror)
                        {
                            // All vectors are stored on every node.
                            bufferSize = totalNumVectors;
                            // Initiate the additional amplitude storage for the reduce
                            // operation and initiate it with zeros.
                            amplitudeMaster.assign(totalNumVectors, complex_X(0.0));
                        }
                        else
                        {
                            countRanks = gc.getGpuNodes().productOfComponents();
                            // Number of scattering vectors in all but last chunk.
                            // (ceil integer division)
                            bufferSize = totalNumVectors / countRanks + ((totalNumVectors % countRanks) != 0);
                            // Number of scattering vectors on the last chunk.
                            resOfVectors = bufferSize - (bufferSize * countRanks - totalNumVectors);
                            // Initiate the additional amplitude storage for receiving
                            // data and initiate it with zeros.
                            amplitudeReceive.assign(bufferSize, complex_X(0.0));
                        }
                        // Allocate amplitude buffer.
                        amplitude = std::make_unique<ComplexBuffer>(DataSpace<DIM1>(bufferSize));
                        // Initialize, on device, its fields with zero.
                        amplitude->getDeviceBuffer().setValue(0.0);

                        // Go to PIC unit system.
                        constexpr float_X invMeterToInvAngstrom = 1.0e10;
                        q_min = q_min * invMeterToInvAngstrom * UNIT_LENGTH;
                        q_max = q_max * invMeterToInvAngstrom * UNIT_LENGTH;
                        // Set the q-space grid spacing.
                        q_step = (q_max - q_min) / precisionCast<float_X>(numVectors);

                        // Rank 0 creates the output directory.
                        pmacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();
                        if(isMaster)
                        {
                            fs.createDirectory("xrayScatteringOutput");
                            fs.setDirectoryPermissions("xrayScatteringOutput");
                        }

                        // Chose the solver for populating a TmpField with the electron
                        // density (either the species density or the bound electron
                        // density).
                        using ElectronDensitySolver = typename DetermineElectronDensitySolver<T_ParticlesType>::type;
                        // Output unit:
                        const float_64 amplitudeUnit
                            = static_cast<float_64>(FieldTmp::getUnit<ElectronDensitySolver>()[0]) * CELL_WIDTH_SI
                            * CELL_HEIGHT_SI * CELL_DEPTH_SI * ELECTRON_RADIUS_SI;

                        // Set the total number of cells in the simulation.
                        totalSimulationCells
                            = Environment<simDim>::get().SubGrid().getGlobalDomain().size.productOfComponents();

                        // Initialize an object responsible for output writing.
                        dataWriter = std::make_unique<XrayScatteringWriter<float_X>>(
                            pluginPrefix + "Output",
                            fileExtension,
                            "xrayScatteringOutput",
                            outputLayout,
                            compressionMethod,
                            precisionCast<uint64_t>(numVectors),
                            q_step,
                            amplitudeUnit,
                            totalSimulationCells);
                        // Set the output period.
                        outputPeriod = pluginSystem::toTimeSlice(outputPeriod_s);
                    }
                }


                void pluginUnload() override
                {
                }


                //! Collect amplitude data from each CPU on the master node.
                void collectIntensityOnMaster()
                {
                    amplitude->deviceToHost();
                    __getTransactionEvent().waitForFinished();

                    reduce(
                        nvidia::functors::Add(),
                        amplitudeMaster.data(),
                        amplitude->getHostBuffer().getBasePointer(),
                        amplitude->getHostBuffer().getCurrentSize(),
                        mpi::reduceMethods::Reduce());
                }


                //! Calculates the offset to the the currently processed output chunk.
                HINLINE uint32_t calcOffset(uint32_t const& step) const
                {
                    /* Chunks move with every "rotation" from left to the right (from
                     * smaller to a higher rank). So after one rotation the rank n has
                     * the n-1 chunk( counted from 0).
                     * so: chunk = (rank - rotations) % countRanks
                     * to avoid a negative number in the modulo operation countRanks
                     * is added in the beginning and only totalRotations % countRanks
                     * is subtracted.
                     */
                    uint32_t totalRotations = accumulatedRotations + step;
                    uint32_t chunk = mpiRank + countRanks;
                    chunk = ((chunk - (totalRotations % countRanks)) % countRanks);
                    return chunk * amplitude->getHostBuffer().getCurrentSize();
                }


                //! Checks if this node hast the last output part.
                HINLINE bool hasLastChunk(uint32_t const& step) const
                {
                    uint32_t totalRotations = accumulatedRotations + step;
                    return mpiRank == (countRanks - 1 + totalRotations) % countRanks;
                }


                //! Writes amplitude data to disk.
                HINLINE void writeOutput()
                {
                    if(outputLayout == OutputMemoryLayout::Distribute)
                    {
                        amplitude->deviceToHost();
                        __getTransactionEvent().waitForFinished();
                        realPart = extractReal(amplitude->getHostBuffer());
                        imgPart = extractImag(amplitude->getHostBuffer());

                        uint64_t offset = precisionCast<uint64_t>(calcOffset(countRanks - 1));
                        uint64_t extent;
                        if(hasLastChunk(countRanks - 1))
                            extent = resOfVectors;
                        else
                            extent = amplitude->getHostBuffer().getCurrentSize();
                        (*dataWriter)(currentStep, extent, offset, realPart, imgPart);
                    }
                    else
                    {
                        collectIntensityOnMaster();
                        if(isMaster)
                        {
                            realPart = extractReal(amplitudeMaster);
                            imgPart = extractImag(amplitudeMaster);
                            (*dataWriter)(currentStep, realPart, imgPart);
                        }
                        // reset amplitudes back to zero
                        amplitudeMaster.assign(amplitudeMaster.size(), complex_X(0.0));
                    }
                }


                /** Passes output chunks from one device to another.
                 *
                 * @param step Current step in the Loop over kernel runs, in the current
                 *     simulation step.
                 */
                HINLINE void communicationOnStep(uint32_t const& step)
                {
                    using namespace mpi;
                    // No action is necessary on the first step.
                    if(step == 0u)
                        return;
                    // Copy data calculated on GPU , on last step, to CPU memory.
                    amplitude->deviceToHost();
                    // Avoid deadlock between not finished pmacc tasks and mpi blocking
                    // collectives.
                    __getTransactionEvent().waitForFinished();
                    // MPI asynchronous send & receive:
                    int bytesToSend = sizeof(complex_X) / sizeof(char);
                    bytesToSend *= amplitude->getHostBuffer().getCurrentSize();

                    // An mpi request to monitor a non blocking send transaction.
                    GridController<simDim>& gc = Environment<simDim>::get().GridController();
                    MPI_Request transactionRequest;
                    // Pass data to the next node.
                    MPI_CHECK(MPI_Isend(
                        amplitude->getHostBuffer().getBasePointer(),
                        bytesToSend,
                        MPI_BYTE,
                        (mpiRank + 1) % countRanks,
                        0,
                        gc.getCommunicator().getMPIComm(),
                        &transactionRequest));
                    // Receive from the proceeding node (blocking transaction).
                    int receiveFrom = (mpiRank == 0u) ? countRanks - 1 : mpiRank - 1;
                    MPI_CHECK(MPI_Recv(
                        amplitudeReceive.data(),
                        bytesToSend,
                        MPI_BYTE,
                        std::move(receiveFrom),
                        0,
                        gc.getCommunicator().getMPIComm(),
                        MPI_STATUS_IGNORE));

                    // Wait for the send transaction to end.
                    MPI_Wait(&transactionRequest, MPI_STATUS_IGNORE);
                    // Copy the received data to the host buffer.
                    copyVectorToBuffer(amplitudeReceive, amplitude->getHostBuffer());
                    // Copy the received data to the device so it can be used as
                    // output in this step.
                    amplitude->hostToDevice();
                }


                /** Calculates a form factor number density of the species.
                 *
                 * @param dc data connector
                 * @param globalOffset offset from the global to the local domain
                 * @return data box containing the calculated data.
                 */
                HINLINE FieldTmp::DataBoxType calculateDensity(DataConnector& dc, DataSpace<simDim>& globalOffset)
                {
                    // Check if there is at least one unused field available.
                    PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
                    // Get a field for density storage.
                    auto tmpField = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                    // Initiate with zeros.
                    tmpField->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType::create(0.0));
                    // Chose species.
                    auto species = dc.get<T_ParticlesType>(T_ParticlesType::FrameType::getName(), true);

                    // Chose the solver for populating a TmpField with the form factor
                    // density of the particles.
                    using ElectronDensitySolver = typename DetermineElectronDensitySolver<T_ParticlesType>::type;
                    // Calculate density.
                    tmpField->template computeValue<CORE + BORDER, ElectronDensitySolver>(*species, currentStep);
                    // Release particle data.
                    dc.releaseData(T_ParticlesType::FrameType::getName());
                    // Get the field data box.
                    FieldTmp::DataBoxType tmpFieldBox = tmpField->getGridBuffer().getDeviceBuffer().getDataBox();
                    return tmpFieldBox;
                }


                /** Runs kernel when the output is distributed over nodes.
                 *
                 * A single kernel run adds result only to that output part which
                 * currently resides on the node. The output parts are passed along to
                 * the neighbouring node, in a circle. This repeats until every node has
                 * computed all scattering vectors.
                 *
                 * @param cellsGrid field grid, without GUARD, on one device
                 * @param fieldTmpNoGuard field data
                 * @param globalOffset offset from the global to the local domain
                 * @param numBlocks number of virtual blocks used in a kernel run
                 * @param fieldPos TmpField in cell position
                 */
                template<typename T_FieldPos>
                HINLINE void runKernelInDistributeMode(
                    DataSpace<simDim>& cellsGrid,
                    FieldTmp::DataBoxType const& fieldTmpNoGuard,
                    DataSpace<simDim>& globalOffset,
                    uint32_t const& numBlocks,
                    T_FieldPos const& fieldPos)
                {
                    // The available number of virtual workers.
                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    // Loop over kernel runs.
                    for(uint32_t step = 0; step < countRanks; step++)
                    {
                        uint32_t countVectors, iterOffset;
                        // Pass along the data.
                        communicationOnStep(step);
                        // 1D offset to the begin of the currently processed output
                        // part.
                        iterOffset = calcOffset(step);
                        // Define scattering vectors for the output part.
                        GetScatteringVector scatteringVectors{q_min, q_max, q_step, numVectors, iterOffset};
                        // Handle possibly smaller amount of vectors to be processed
                        // in the last output part.
                        if(hasLastChunk(step))
                        {
                            countVectors = resOfVectors;
                        }
                        else
                            countVectors = amplitude->getHostBuffer().getCurrentSize();
                        // Start the kernel.
                        PMACC_KERNEL(KernelXrayScattering<numWorkers>{})
                        (numBlocks, numWorkers)(
                            cellsGrid,
                            fieldTmpNoGuard,
                            globalOffset,
                            fieldPos,
                            amplitude->getDeviceBuffer().getDataBox(),
                            countVectors,
                            scatteringVectors,
                            *probingBeam,
                            currentStep,
                            totalSimulationCells);
                    }
                }

                /** Runs xrayScattering kernel when the output is mirrored over nodes.
                 *
                 * Kernel runs only once in a simulation time step and computes
                 * the complete output at once.
                 *
                 * @param cellsGrid field grid, without GUARD, on one device
                 * @param fieldTmpNoGuard field data
                 * @param globalOffset offset from the global to the local domain
                 * @param numBlocks number of virtual blocks used in a kernel run
                 * @param fieldPos TmpField in cell position
                 */
                template<typename T_FieldPos>
                HINLINE void runKernelInMirrorMode(
                    DataSpace<simDim>& cellsGrid,
                    FieldTmp::DataBoxType const& fieldTmpNoGuard,
                    DataSpace<simDim>& globalOffset,
                    uint32_t const& numBlocks,
                    T_FieldPos const& fieldPos)
                {
                    // Get the available number of virtual workers.
                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                    // Define scattering vectors for the output part.
                    GetScatteringVector scatteringVectors{q_min, q_max, q_step, numVectors, 0};
                    // Run the kernel.
                    PMACC_KERNEL(KernelXrayScattering<numWorkers>{})
                    (numBlocks, numWorkers)(
                        cellsGrid,
                        fieldTmpNoGuard,
                        globalOffset,
                        fieldPos,
                        amplitude->getDeviceBuffer().getDataBox(),
                        amplitude->getHostBuffer().getCurrentSize(),
                        scatteringVectors,
                        *probingBeam,
                        currentStep,
                        totalSimulationCells);
                }


                /** Actions performed on every step included in the notify period.
                 *
                 * First the form factor density is calculated then the Kernel is
                 * started. For steps in the output period, amplitude is written
                 * to disk.
                 *
                 * @param currentStep
                 **/
                HINLINE void notify(uint32_t currentStep) override
                {
                    this->currentStep = currentStep;

                    // Get the available number of virtual workers per block.
                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    // Form factor density:
                    // Get the offset to the local domain (this HOST + DEVICE pair).
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
                    // Calculate the density and get a data box to access this TmpField.
                    DataConnector& dc = Environment<>::get().DataConnector();
                    FieldTmp::DataBoxType tmpFieldBox = calculateDensity(dc, globalOffset);
                    // Get the in cell position of a TmpField.
                    // Could probably remove it as it is the cell origin in all cell
                    // types.
                    const picongpu::traits::FieldPosition<typename fields::CellType, FieldTmp> fieldPos;
                    // Shift the density box to exclude the GUARD.
                    DataSpace<simDim> guardingSC = cellDescription.getGuardingSuperCells();
                    auto const fieldTmpNoGuard = tmpFieldBox.shift(guardingSC * SuperCellSize::toRT());
                    // Get the field size on this rank (no GUARD).
                    DataSpace<simDim> cellsGrid
                        = (cellDescription.getGridSuperCells() - 2 * guardingSC) * SuperCellSize::toRT();
                    uint32_t const totalNumCells = cellsGrid.productOfComponents();
                    // Get the number of, virtual, blocks.
                    PMACC_ASSERT(totalNumCells % numWorkers == 0);
                    uint32_t const numBlocks = totalNumCells / numWorkers;


                    // Run Kernel.
                    if(outputLayout == OutputMemoryLayout::Distribute)
                    {
                        runKernelInDistributeMode(cellsGrid, fieldTmpNoGuard, globalOffset, numBlocks, fieldPos);
                    }
                    else
                    {
                        runKernelInMirrorMode(cellsGrid, fieldTmpNoGuard, globalOffset, numBlocks, fieldPos);
                    }
                    // Release density data.
                    dc.releaseData(FieldTmp::getUniqueId(0));
                    // Write to disk.
                    if(pluginSystem::containsStep(outputPeriod, currentStep))
                        writeOutput();
                    // Update the total number of rotations ( data passes ).
                    if(outputLayout == OutputMemoryLayout::Distribute)
                        accumulatedRotations += countRanks - 1;
                }
            };
        } // namespace xrayScattering
    } // namespace plugins
    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, plugins::xrayScattering::XrayScattering<T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                // This plugin needs at least the position and weighting.
                using RequiredIdentifiers = MakeSeq_t<position<>, weighting>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                using type = SpeciesHasIdentifiers;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
