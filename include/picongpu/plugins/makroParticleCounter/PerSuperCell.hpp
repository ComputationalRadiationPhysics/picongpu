/* Copyright 2014-2021 Rene Widera, Richard Pausch
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

#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include "picongpu/plugins/ILightweightPlugin.hpp"

#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <splash/splash.h>

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>


namespace picongpu
{
    using namespace pmacc;
    using namespace splash;

    struct CountMakroParticle
    {
        template<typename ParBox, typename CounterBox, typename Mapping, typename T_Acc>
        DINLINE void operator()(T_Acc const& acc, ParBox parBox, CounterBox counterBox, Mapping mapper) const
        {
            typedef MappingDesc::SuperCellSize SuperCellSize;
            typedef typename ParBox::FrameType FrameType;
            typedef typename ParBox::FramePtr FramePtr;

            const DataSpace<simDim> block(mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));
            /* counterBox has no guarding supercells*/
            const DataSpace<simDim> counterCell = block - mapper.getGuardingSuperCells();

            const DataSpace<simDim> threadIndex(cupla::threadIdx(acc));
            const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize>(threadIndex);

            PMACC_SMEM(acc, counterValue, uint64_cu);
            PMACC_SMEM(acc, frame, FramePtr);

            if(linearThreadIdx == 0)
            {
                counterValue = 0;
                frame = parBox.getLastFrame(block);
                if(!frame.isValid())
                {
                    counterBox(counterCell) = counterValue;
                }
            }
            cupla::__syncthreads(acc);
            if(!frame.isValid())
                return; // end kernel if we have no frames

            bool isParticle = frame[linearThreadIdx][multiMask_];

            while(frame.isValid())
            {
                if(isParticle)
                {
                    cupla::atomicAdd(acc, &counterValue, static_cast<uint64_cu>(1LU), ::alpaka::hierarchy::Blocks{});
                }
                cupla::__syncthreads(acc);
                if(linearThreadIdx == 0)
                {
                    frame = parBox.getPreviousFrame(frame);
                }
                isParticle = true;
                cupla::__syncthreads(acc);
            }

            if(linearThreadIdx == 0)
                counterBox(counterCell) = counterValue;
        }
    };
    /** Count makro particle of a species and write down the result to a global HDF5 file.
     *
     * - count the total number of makro particle per supercell
     * - store one number (size_t) per supercell in a mesh
     * - Output: - create a folder with the name of the plugin
     *           - per time step one file with the name "result_[currentStep].h5" is created
     * - HDF5 Format: - default lib splash output for meshes
     *                - the attribute name in the HDF5 file is "makroParticleCount"
     *
     */
    template<class ParticlesType>
    class PerSuperCell : public ILightweightPlugin
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;
        typedef GridBuffer<size_t, simDim> GridBufferType;

        MappingDesc* cellDescription;
        std::string notifyPeriod;

        std::string pluginName;
        std::string pluginPrefix;
        std::string foldername;
        mpi::MPIReduce reduce;

        GridBufferType* localResult;

        ParallelDomainCollector* dataCollector;
        // set attributes for datacollector files
        DataCollector::FileCreationAttr h5_attr;

    public:
        PerSuperCell()
            : pluginName("PerSuperCell: create hdf5 with macro particle count per superCell")
            , pluginPrefix(ParticlesType::FrameType::getName() + std::string("_macroParticlesPerSuperCell"))
            , foldername(pluginPrefix)
            , cellDescription(nullptr)
            , localResult(nullptr)
            , dataCollector(nullptr)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PerSuperCell()
        {
        }

        void notify(uint32_t currentStep)
        {
            countMakroParticles<CORE + BORDER>(currentStep);
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad()
        {
            if(!notifyPeriod.empty())
            {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                /* local count of supercells without any guards*/
                DataSpace<simDim> localSuperCells(subGrid.getLocalDomain().size / SuperCellSize::toRT());
                localResult = new GridBufferType(localSuperCells);

                /* create folder for hdf5 files*/
                Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(foldername);
            }
        }

        void pluginUnload()
        {
            __delete(localResult);

            if(dataCollector)
                dataCollector->finalize();

            __delete(dataCollector);
        }

        template<uint32_t AREA>
        void countMakroParticles(uint32_t currentStep)
        {
            openH5File();

            DataConnector& dc = Environment<>::get().DataConnector();

            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);

            /*############ count particles #######################################*/
            typedef MappingDesc::SuperCellSize SuperCellSize;
            AreaMapping<AREA, MappingDesc> mapper(*cellDescription);

            PMACC_KERNEL(CountMakroParticle{})
            (mapper.getGridDim(), SuperCellSize::toRT())(
                particles->getDeviceParticlesBox(),
                localResult->getDeviceBuffer().getDataBox(),
                mapper);

            dc.releaseData(ParticlesType::FrameType::getName());

            localResult->deviceToHost();


            /*############ dump data #############################################*/
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

            DataSpace<simDim> localSize(subGrid.getLocalDomain().size / SuperCellSize::toRT());
            DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset / SuperCellSize::toRT());
            DataSpace<simDim> globalSize(subGrid.getGlobalDomain().size / SuperCellSize::toRT());


            Dimensions splashGlobalDomainOffset(0, 0, 0);
            Dimensions splashGlobalOffset(0, 0, 0);
            Dimensions splashGlobalDomainSize(1, 1, 1);
            Dimensions splashGlobalSize(1, 1, 1);
            Dimensions localBufferSize(1, 1, 1);

            for(uint32_t d = 0; d < simDim; ++d)
            {
                splashGlobalOffset[d] = globalOffset[d];
                splashGlobalSize[d] = globalSize[d];
                splashGlobalDomainSize[d] = globalSize[d];
                localBufferSize[d] = localSize[d];
            }

            size_t* ptr = localResult->getHostBuffer().getPointer();

            // avoid deadlock between not finished pmacc tasks and mpi calls in adios
            __getTransactionEvent().waitForFinished();

            dataCollector->writeDomain(
                currentStep, /* id == time step */
                splashGlobalSize, /* total size of dataset over all processes */
                splashGlobalOffset, /* write offset for this process */
                ColTypeUInt64(), /* data type */
                simDim, /* NDims of the field data (scalar, vector, ...) */
                splash::Selection(localBufferSize),
                "makroParticlePerSupercell", /* data set name */
                splash::Domain(
                    splashGlobalDomainOffset, /* offset of the global domain */
                    splashGlobalDomainSize /* size of the global domain */
                    ),
                DomainCollector::GridType,
                ptr);

            closeH5File();
        }

        void closeH5File()
        {
            if(dataCollector != nullptr)
            {
                std::string filename = (foldername + std::string("/makroParticlePerSupercell"));
                log<picLog::INPUT_OUTPUT>("HDF5 close DataCollector with file: %1%") % filename;
                dataCollector->close();
            }
        }

        void openH5File()
        {
            if(dataCollector == nullptr)
            {
                DataSpace<simDim> mpi_pos;
                DataSpace<simDim> mpi_size;

                Dimensions splashMpiPos;
                Dimensions splashMpiSize;

                GridController<simDim>& gc = Environment<simDim>::get().GridController();

                mpi_pos = gc.getPosition();
                mpi_size = gc.getGpuNodes();

                splashMpiPos.set(0, 0, 0);
                splashMpiSize.set(1, 1, 1);

                for(uint32_t i = 0; i < simDim; ++i)
                {
                    splashMpiPos[i] = mpi_pos[i];
                    splashMpiSize[i] = mpi_size[i];
                }


                const uint32_t maxOpenFilesPerNode = 1;
                dataCollector = new ParallelDomainCollector(
                    gc.getCommunicator().getMPIComm(),
                    gc.getCommunicator().getMPIInfo(),
                    splashMpiSize,
                    maxOpenFilesPerNode);
                // set attributes for datacollector files
                DataCollector::FileCreationAttr h5_attr;
                h5_attr.enableCompression = false;
                h5_attr.fileAccType = DataCollector::FAT_CREATE;
                h5_attr.mpiPosition.set(splashMpiPos);
                h5_attr.mpiSize.set(splashMpiSize);
            }


            // open datacollector
            try
            {
                std::string filename = (foldername + std::string("/makroParticlePerSupercell"));
                log<picLog::INPUT_OUTPUT>("HDF5 open DataCollector with file: %1%") % filename;
                dataCollector->open(filename.c_str(), h5_attr);
            }
            catch(const DCException& e)
            {
                std::cerr << e.what() << std::endl;
                throw std::runtime_error("Failed to open datacollector");
            }
        }
    };

} // namespace picongpu
