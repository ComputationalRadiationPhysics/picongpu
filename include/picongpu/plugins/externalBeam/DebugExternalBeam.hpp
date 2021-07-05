/* Copyright 2021 Pawel Ordyna
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

#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/common/openPMDAttributes.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/openPMDWriteMeta.hpp"
#include "picongpu/plugins/externalBeam/ProbingBeam.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <openPMD/openPMD.hpp>


namespace picongpu
{
    namespace plugins
    {
        namespace externalBeam
        {
            using namespace pmacc;

            /* Kernel for DebugExternalBeam
             *
             * Fill a grid buffer with beam intensity. The beam ts defined by an `externalBeam::ProbingBeam` instance.
             * @tparam T_numWorkers number of virtual workers for lockstep execution.
             */
            template<uint32_t T_numWorkers>
            struct FillIntensityBuffer
            {
                /** Kernel operator
                 *
                 * @param acc alpaka accelerator
                 * @param intensityBox Device side data box of the output Buffer.
                 * @param globalOffset Offset from the global to the local domain.
                 * @param mapper kernel mapping description. Maps kernel instance to a super cells.
                 * @param probingBeam Probing beam characterization.
                 * @param currentStep Current simulation step.
                 */
                template<typename T_Acc, typename T_IntensityBox, typename T_Mapping, typename T_ProbingBeam>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_IntensityBox intensityBox,
                    DataSpace<simDim> globalOffset,
                    T_Mapping mapper,
                    T_ProbingBeam probingBeam,
                    uint32_t currentStep) const
                {
                    constexpr uint32_t numWorkers = T_numWorkers;
                    typedef MappingDesc::SuperCellSize SuperCellSize;
                    constexpr uint32_t cellsPerSupercell = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    uint32_t const workerIdx = cupla::threadIdx(acc).x;

                    const DataSpace<simDim> block(mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));
                    // counterBox has no guarding super cells
                    const DataSpace<simDim> blockCell = block - mapper.getGuardingSuperCells();
                    auto intensityBoxBlock = intensityBox.shift(blockCell * SuperCellSize::toRT());
                    auto forEachCellInSuperCell = lockstep::makeForEach<cellsPerSupercell, numWorkers>(workerIdx);
                    forEachCellInSuperCell(
                        [&](lockstep::Idx const idx)
                        {
                            DataSpace<simDim> const cellPosition(
                                DataSpaceOperations<simDim>::map(SuperCellSize::toRT(), idx));
                            DataSpace<simDim> const cellGlobalPosition{
                                cellPosition + blockCell * SuperCellSize::toRT() + globalOffset};
                            floatD_X fieldGlobalPosition
                                = precisionCast<float_X>(cellGlobalPosition) + float3_X::create(0.5_X);
                            fieldGlobalPosition *= cellSize.shrink<simDim>();
                            float3_X position_b = probingBeam.coordinateTransform(currentStep, fieldGlobalPosition);
                            float_X intensity = probingBeam(position_b);
                            intensityBoxBlock(cellPosition) = intensity;
                        });
                }
            };
            /** Output external Beam intensity for each simulation cell
             *
             * - call the probing beam functor defined in debugExternalBeam.param for each simulation cell
             * - store the resulting beam intensities
             * - Output: - create a folder with the name of the plugin
             *           - per time step one file with the name "debugExternalBeam_[currentStep].h5" is created
             *             (or a different extension in case of another openPMD backend)
             * - HDF5 Format: - default openPMD output for meshes
             *                - the attribute name in the HDF5 file is "externalBeamIntensity"
             *
             */
            class DebugExternalBeam : public ILightweightPlugin
            {
            private:
                typedef MappingDesc::SuperCellSize SuperCellSize;
                typedef GridBuffer<float_X, simDim> GridBufferType;

                MappingDesc cellDescription_m;
                std::string notifyPeriod;
                std::string filenameExtension_m;
                std::string filenameInfix_m;

                std::string pluginName_m;
                std::string pluginPrefix_m;
                std::string dirName_m;

                std::unique_ptr<pmacc::traits::Resolve<DebugBeam>::type> probingBeam;
                std::unique_ptr<GridBufferType> localResult_m;

                // @todo upon switching to C++17, use std::option instead
                std::unique_ptr<::openPMD::Series> series_m;
                // set attributes for datacollector files

            public:
                HINLINE DebugExternalBeam()
                    : pluginName_m("DebugExternalBeam: output external Beam intensity for each simulation cell")
                    , pluginPrefix_m(std::string("debugExternalBeam"))
                    , dirName_m(pluginPrefix_m)
                    , cellDescription_m(DataSpace<simDim>(SuperCellSize::toRT()))
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                HINLINE ~DebugExternalBeam() override
                {
                }

                HINLINE void notify(uint32_t currentStep) override
                {
                    fillLocalBuffer(currentStep);
                    writeData(currentStep);
                }

                HINLINE void pluginRegisterHelp(po::options_description& desc) override
                {
                    desc.add_options()(
                        (pluginPrefix_m + ".period").c_str(),
                        po::value<std::string>(&notifyPeriod),
                        "enable plugin [for each n-th step]");
                    desc.add_options()(
                        (pluginPrefix_m + ".ext").c_str(),
                        po::value<std::string>(&filenameExtension_m)->default_value("h5"),
                        "openPMD filename extension (default: 'h5')");
                    desc.add_options()(
                        (pluginPrefix_m + ".infix").c_str(),
                        po::value<std::string>(&filenameInfix_m)->default_value("_%06T"),
                        "openPMD filename infix (default: '_%06T' for file-based iteration layout, pick 'NULL' for "
                        "group-based layout");
                }

                HINLINE std::string pluginGetName() const override
                {
                    return pluginName_m;
                }

                HINLINE void setMappingDescription(MappingDesc* cellDescription) override
                {
                    cellDescription_m = *cellDescription;
                }

            private:
                HINLINE void pluginLoad() override
                {
                    if(!notifyPeriod.empty())
                    {
                        probingBeam = std::make_unique<DebugBeam>();
                        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
                        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> localCells(subGrid.getLocalDomain().size);
                        localResult_m = std::make_unique<GridBufferType>(localCells);
                        /* create folder for output files*/
                        Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(dirName_m);
                    }
                }

                HINLINE void pluginUnload() override
                {
                    series_m.reset();
                }

                HINLINE void fillLocalBuffer(uint32_t currentStep) const
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                    typedef MappingDesc::SuperCellSize SuperCellSize;
                    AreaMapping<CORE + BORDER, MappingDesc> mapper(cellDescription_m);

                    // Get the field size on this rank (no GUARD).
                    DataSpace<simDim> localDomainOffset(subGrid.getLocalDomain().offset);
                    PMACC_KERNEL(FillIntensityBuffer<numWorkers>{})
                    (mapper.getGridDim(), numWorkers)(
                        localResult_m->getDeviceBuffer().getDataBox(),
                        localDomainOffset,
                        mapper,
                        *probingBeam,
                        currentStep);

                    localResult_m->deviceToHost();
                }
                HINLINE void writeData(uint32_t currentStep)
                {
                    openSeries();
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                    DataSpace<simDim> localDomainSize(subGrid.getLocalDomain().size);
                    DataSpace<simDim> localDomainOffset(subGrid.getLocalDomain().offset);
                    DataSpace<simDim> globalDomainSize(subGrid.getGlobalDomain().size);

                    ::openPMD::Extent openPmdGlobalDomainExtent(simDim);
                    ::openPMD::Extent openPmdLocalDomainOffset(simDim);
                    ::openPMD::Offset openPmdLocalDomainExtent(simDim);

                    for(::openPMD::Extent::value_type d = 0; d < simDim; ++d)
                    {
                        openPmdGlobalDomainExtent[simDim - d - 1] = globalDomainSize[d];
                        openPmdLocalDomainOffset[simDim - d - 1] = localDomainOffset[d];
                        openPmdLocalDomainExtent[simDim - d - 1] = localDomainSize[d];
                    }

                    float_X* ptr = localResult_m->getHostBuffer().getPointer();

                    // avoid deadlock between not finished pmacc tasks and mpi calls in adios
                    __getTransactionEvent().waitForFinished();

                    auto iteration = series_m->WRITE_ITERATIONS[currentStep];
                    auto mesh = iteration.meshes["externalBeamIntensity"];
                    auto dataset = mesh[::openPMD::RecordComponent::SCALAR];
                    openPMD::SetMeshAttributes setMeshAttributes(currentStep);
                    setMeshAttributes(mesh)(dataset);
                    dataset.resetDataset({::openPMD::determineDatatype<float_X>(), openPmdGlobalDomainExtent});
                    dataset.storeChunk(::openPMD::shareRaw(ptr), openPmdLocalDomainOffset, openPmdLocalDomainExtent);

                    openPMD::WriteMeta writeMetaAttributes;
                    writeMetaAttributes(
                        *series_m,
                        currentStep,
                        /* writeFieldMeta = */ false,
                        /* writeParticleMeta = */ false,
                        /* writeToLog = */ false);

                    iteration.close();
                }

                HINLINE void openSeries()
                {
                    if(!series_m)
                    {
                        GridController<simDim>& gc = Environment<simDim>::get().GridController();

                        std::string infix = filenameInfix_m;
                        if(infix == "NULL")
                        {
                            infix = "";
                        }
                        std::string filename = dirName_m + std::string("/debugExternalBeam") + infix + std::string(".")
                            + filenameExtension_m;
                        log<picLog::INPUT_OUTPUT>("openPMD open Series at: %1%") % filename;

                        series_m = std::make_unique<::openPMD::Series>(
                            filename,
                            ::openPMD::Access::CREATE,
                            gc.getCommunicator().getMPIComm());
                    }
                }
            };
        } // namespace externalBeam
    } // namespace plugins
} // namespace picongpu
