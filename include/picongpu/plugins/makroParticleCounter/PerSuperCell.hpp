/* Copyright 2014-2023 Rene Widera, Richard Pausch
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
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#include "picongpu/plugins/common/openPMDWriteMeta.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <openPMD/openPMD.hpp>


namespace picongpu
{
    using namespace pmacc;

    //! Count macro particles per superCell
    struct CountMakroParticle
    {
        template<typename ParBox, typename CounterBox, typename Mapping, typename T_Worker>
        DINLINE void operator()(T_Worker const& worker, ParBox parBox, CounterBox counterBox, Mapping mapper) const
        {
            const DataSpace<simDim> superCellIdx(
                mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));
            /* counterBox has no guarding supercells*/
            const DataSpace<simDim> superCellIdxNoGuard = superCellIdx - mapper.getGuardingSuperCells();

            PMACC_SMEM(worker, counterValue, uint64_cu);

            auto masterOnly = lockstep::makeMaster(worker);

            masterOnly([&]() { counterValue = 0; });
            worker.sync();

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, parBox, superCellIdx);

            forEachParticle(
                [&counterValue](auto const& lockstepWorker, auto& /*particle*/)
                {
                    cupla::atomicAdd(
                        lockstepWorker.getAcc(),
                        &counterValue,
                        static_cast<uint64_cu>(1LU),
                        ::alpaka::hierarchy::Threads{});
                });

            worker.sync();

            masterOnly(
                [&]()
                {
                    PMACC_DEVICE_ASSERT_MSG(
                        counterValue == forEachParticle.numParticles(),
                        "[makroParticlesCounter] Number of particles counted and given by the iteration algorithm "
                        "differ.");
                    counterBox(superCellIdxNoGuard) = counterValue;
                });
        }
    };
    /** Count makro particle of a species and write down the result to a global HDF5 file.
     *
     * - count the total number of makro particle per supercell
     * - store one number (size_t) per supercell in a mesh
     * - Output: - create a folder with the name of the plugin
     *           - per time step one file with the name "result_[currentStep].h5" is created
     *             (or a different extension in case of another openPMD backend)
     * - HDF5 Format: - default openPMD output for meshes
     *                - the attribute name in the HDF5 file is "makroParticleCount"
     *
     */
    template<class ParticlesType>
    class PerSuperCell : public ILightweightPlugin
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;
        typedef GridBuffer<size_t, simDim> GridBufferType;

        std::string notifyPeriod;
        std::string m_filenameExtension = openPMD::getDefaultExtension();
        std::string m_filenameInfix = "_%06T";

        std::string pluginName;
        std::string pluginPrefix;
        std::string foldername;
        MappingDesc* cellDescription;

        mpi::MPIReduce reduce;

        std::unique_ptr<GridBufferType> localResult;

        // @todo upon switching to C++17, use std::option instead
        std::unique_ptr<::openPMD::Series> m_Series;

        ::openPMD::Offset m_offset;
        ::openPMD::Extent m_extent;

    public:
        PerSuperCell()
            : pluginName("PerSuperCell: create hdf5 with macro particle count per superCell")
            , pluginPrefix(ParticlesType::FrameType::getName() + std::string("_macroParticlesPerSuperCell"))
            , foldername(pluginPrefix)
            , cellDescription(nullptr)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PerSuperCell()
        {
        }

        void notify(uint32_t currentStep) override
        {
            countMakroParticles<CORE + BORDER>(currentStep);
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
            desc.add_options()(
                (pluginPrefix + ".ext").c_str(),
                po::value<std::string>(&m_filenameExtension),
                ("openPMD filename extension (default: '" + m_filenameExtension + "')").c_str());
            desc.add_options()(
                (pluginPrefix + ".infix").c_str(),
                po::value<std::string>(&m_filenameInfix),
                "openPMD filename infix (default: '_%06T' for file-based iteration layout, pick 'NULL' for "
                "group-based layout");
        }

        std::string pluginGetName() const override
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad() override
        {
            if(!notifyPeriod.empty())
            {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                /* local count of supercells without any guards*/
                DataSpace<simDim> localSuperCells(subGrid.getLocalDomain().size / SuperCellSize::toRT());
                localResult = std::make_unique<GridBufferType>(localSuperCells);

                /* create folder for hdf5 files*/
                Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(foldername);
            }
        }

        void pluginUnload() override
        {
            m_Series.reset();
        }

        template<uint32_t AREA>
        void countMakroParticles(uint32_t currentStep)
        {
            openSeries();

            DataConnector& dc = Environment<>::get().DataConnector();

            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            /*############ count particles #######################################*/
            using SuperCellSize = MappingDesc::SuperCellSize;
            auto const mapper = makeAreaMapper<AREA>(*cellDescription);

            auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();
            PMACC_LOCKSTEP_KERNEL(CountMakroParticle{}, workerCfg)
            (mapper.getGridDim())(
                particles->getDeviceParticlesBox(),
                localResult->getDeviceBuffer().getDataBox(),
                mapper);

            localResult->deviceToHost();


            /*############ dump data #############################################*/
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

            DataSpace<simDim> localDomainSize(subGrid.getLocalDomain().size / SuperCellSize::toRT());
            DataSpace<simDim> localDomainOffset(subGrid.getLocalDomain().offset / SuperCellSize::toRT());
            DataSpace<simDim> globalDomainSize(subGrid.getGlobalDomain().size / SuperCellSize::toRT());

            ::openPMD::Extent openPmdGlobalDomainExtent(simDim);

            ::openPMD::Extent openPmdLocalDomainOffset(simDim);
            ::openPMD::Offset openPmdLocalDomainExtent(simDim);

            for(::openPMD::Extent::value_type d = 0; d < simDim; ++d)
            {
                openPmdGlobalDomainExtent[simDim - d - 1] = globalDomainSize[d];
                openPmdLocalDomainOffset[simDim - d - 1] = localDomainOffset[d];
                openPmdLocalDomainExtent[simDim - d - 1] = localDomainSize[d];
            }

            size_t* ptr = localResult->getHostBuffer().getPointer();

            // avoid deadlock between not finished pmacc tasks and collective or blocking MPI calls in openPMD
            eventSystem::getTransactionEvent().waitForFinished();

            auto iteration = m_Series->writeIterations()[currentStep];

            auto mesh = iteration.meshes["makroParticlePerSupercell"];
            auto dataset = mesh[::openPMD::RecordComponent::SCALAR];

            openPMD::SetMeshAttributes setMeshAttributes(currentStep);
            // gridSpacing = SuperCellSize::toRT() * cellSize
            // m_gridSpacing is initialized by the cellSize
            {
                auto superCellSize = SuperCellSize::toRT();
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    setMeshAttributes.m_gridSpacing[simDim - d - 1] *= superCellSize[d];
                }
            }

            setMeshAttributes(mesh)(dataset);

            dataset.resetDataset({::openPMD::determineDatatype<size_t>(), openPmdGlobalDomainExtent});
            dataset.storeChunk(
                std::shared_ptr<size_t>{ptr, [](auto const*) {}},
                openPmdLocalDomainOffset,
                openPmdLocalDomainExtent);

            openPMD::WriteMeta writeMetaAttributes;
            writeMetaAttributes(
                *m_Series,
                iteration,
                currentStep,
                /* writeFieldMeta = */ false,
                /* writeParticleMeta = */ false,
                /* writeToLog = */ false);

            iteration.close();
        }

        void openSeries()
        {
            if(!m_Series)
            {
                GridController<simDim>& gc = Environment<simDim>::get().GridController();

                std::string infix = m_filenameInfix;
                if(infix == "NULL")
                {
                    infix = "";
                }
                std::string filename = foldername + std::string("/makroParticlePerSupercell") + infix
                    + std::string(".") + m_filenameExtension;
                log<picLog::INPUT_OUTPUT>("openPMD open Series at: %1%") % filename;

                m_Series = std::make_unique<::openPMD::Series>(
                    filename,
                    ::openPMD::Access::CREATE,
                    gc.getCommunicator().getMPIComm());
            }
        }
    };

} // namespace picongpu
