/**
 * Copyright 2013-2016 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include <pthread.h>
#include <sstream>
#include <list>
#include <vector>

#include "simulation_defines.hpp"

#include "plugins/hdf5/HDF5Writer.def"
#include "traits/SplashToPIC.hpp"
#include "traits/PICToSplash.hpp"

#include "particles/frame_types.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"
#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/operations/CountParticles.hpp"
#include "particles/IdProvider.def"

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/GridLayout.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "math/Vector.hpp"

#include "plugins/ISimulationPlugin.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>

#include <boost/type_traits.hpp>

#include "plugins/hdf5/WriteFields.hpp"
#include "plugins/hdf5/WriteSpecies.hpp"
#include "plugins/hdf5/restart/LoadSpecies.hpp"
#include "plugins/hdf5/restart/RestartFieldLoader.hpp"
#include "plugins/hdf5/NDScalars.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"

namespace picongpu
{

namespace hdf5
{

using namespace PMacc;

using namespace splash;


namespace po = boost::program_options;

/**
 * Writes simulation data to hdf5 files using libSplash.
 * Implements the ISimulationPlugin interface.
 */
class HDF5Writer : public ISimulationPlugin
{
public:

    HDF5Writer() :
    filename("h5_data"),
    checkpointFilename("h5_checkpoint"),
    restartFilename(""), /* set to checkpointFilename by default */
    notifyPeriod(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~HDF5Writer()
    {

    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("hdf5.period", po::value<uint32_t > (&notifyPeriod)->default_value(0),
             "enable HDF5 IO [for each n-th step]")
            ("hdf5.file", po::value<std::string > (&filename)->default_value(filename),
             "HDF5 output filename (prefix)")
            ("hdf5.checkpoint-file", po::value<std::string > (&checkpointFilename),
             "Optional HDF5 checkpoint filename (prefix)")
            ("hdf5.restart-file", po::value<std::string > (&restartFilename),
             "HDF5 restart filename (prefix)")
            /* 1,000,000 particles are around 3900 frames at 256 particles per frame
             * and match ~30MiB with typical picongpu particles.
             * The only reason why we use 1M particles per chunk is that we can get a
             * frame overflow in our memory manager if we process all particles in one kernel.
             **/
            ("hdf5.restart-chunkSize", po::value<uint32_t > (&restartChunkSize)->default_value(1000000),
             "Number of particles processed in one kernel call during restart to prevent frame count blowup");
    }

    std::string pluginGetName() const
    {
        return "HDF5Writer";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {

        this->cellDescription = cellDescription;
        mThreadParams.cellDescription = this->cellDescription;
    }

    __host__ void notify(uint32_t currentStep)
    {
        notificationReceived(currentStep, false);
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
#if(ENABLE_ADIOS == 1)
        log<picLog::INPUT_OUTPUT > ("HDF5: Checkpoint skipped since ADIOS is enabled.");
#else
        this->checkpointDirectory = checkpointDirectory;

        notificationReceived(currentStep, true);
#endif
    }

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
#if(ENABLE_ADIOS == 1)
        log<picLog::INPUT_OUTPUT > ("HDF5: Restart skipped since ADIOS is enabled.");
#else
        const uint32_t maxOpenFilesPerNode = 4;
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        mThreadParams.dataCollector = new ParallelDomainCollector(
                                                                  gc.getCommunicator().getMPIComm(),
                                                                  gc.getCommunicator().getMPIInfo(),
                                                                  splashMpiSize,
                                                                  maxOpenFilesPerNode);

        mThreadParams.currentStep = restartStep;

        /* set attributes for datacollector files */
        DataCollector::FileCreationAttr attr;
        attr.fileAccType = DataCollector::FAT_READ;
        attr.mpiPosition.set(splashMpiPos);
        attr.mpiSize.set(splashMpiSize);

        /* if restartFilename is relative, prepend with restartDirectory */
        if (!boost::filesystem::path(restartFilename).has_root_path())
        {
            restartFilename = restartDirectory + std::string("/") + restartFilename;
        }

        /* open datacollector */
        try
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % restartFilename;
            mThreadParams.dataCollector->open(restartFilename.c_str(), attr);
        }
        catch (const DCException& e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }

        /* load number of slides to initialize MovingWindow */
        uint32_t slides = 0;
        mThreadParams.dataCollector->readAttribute(restartStep, NULL, "sim_slides", &slides);

        /* apply slides to set gpus to last/written configuration */
        log<picLog::INPUT_OUTPUT > ("Setting slide count for moving window to %1%") % slides;
        MovingWindow::getInstance().setSlideCounter(slides, restartStep);

        /* re-distribute the local offsets in y-direction */
        if( MovingWindow::getInstance().isSlidingWindowActive() )
            gc.setStateAfterSlides(slides);

        /* set window for restart, complete global domain */
        mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(restartStep);
        for (uint32_t i = 0; i < simDim; ++i)
        {
            mThreadParams.localWindowToDomainOffset[i] = 0;
        }

        ThreadParams *params = &mThreadParams;

        /* load all fields */
        ForEach<FileCheckpointFields, LoadFields<bmpl::_1> > forEachLoadFields;
        forEachLoadFields(params);

        /* load all particles */
        ForEach<FileCheckpointParticles, LoadSpecies<bmpl::_1> > forEachLoadSpecies;
        forEachLoadSpecies(params, restartChunkSize);

        uint64_t idProviderNextId, idProviderMaxNumProc;
        ReadNDScalars<uint64_t, uint64_t>()(mThreadParams,
                "picongpu/idProviderState", &idProviderNextId,
                "maxNumProc", &idProviderMaxNumProc);
        log<picLog::INPUT_OUTPUT > ("Setting id on current rank: %1%") % idProviderNextId;
        IdProvider<simDim>::setState(idProviderNextId, idProviderMaxNumProc);

        /* close datacollector */
        log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector with file: %1%") % restartFilename;
        mThreadParams.dataCollector->close();

        if (mThreadParams.dataCollector)
            mThreadParams.dataCollector->finalize();

        __delete(mThreadParams.dataCollector);
#endif
    }

private:

    void closeH5File()
    {
        if (mThreadParams.dataCollector != NULL)
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector");
            mThreadParams.dataCollector->close();
        }
    }

    void openH5File(const std::string h5Filename)
    {
        const uint32_t maxOpenFilesPerNode = 4;
        if (mThreadParams.dataCollector == NULL)
        {
            GridController<simDim> &gc = Environment<simDim>::get().GridController();
            mThreadParams.dataCollector = new ParallelDomainCollector(
                                                                      gc.getCommunicator().getMPIComm(),
                                                                      gc.getCommunicator().getMPIInfo(),
                                                                      splashMpiSize,
                                                                      maxOpenFilesPerNode);
        }
        // set attributes for datacollector files
        DataCollector::FileCreationAttr attr;
        attr.enableCompression = false;
        attr.fileAccType = DataCollector::FAT_CREATE;
        attr.mpiPosition.set(splashMpiPos);
        attr.mpiSize.set(splashMpiSize);

        // open datacollector
        try
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % h5Filename;
            mThreadParams.dataCollector->open(h5Filename.c_str(), attr);
        }
        catch (const DCException& e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }

    }

    /**
     * Notification for dump or checkpoint received
     *
     * @param currentStep current simulation step
     * @param isCheckpoint checkpoint notification
     */
    void notificationReceived(uint32_t currentStep, bool isCheckpoint)
    {
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        mThreadParams.isCheckpoint = isCheckpoint;
        mThreadParams.currentStep = currentStep;
        mThreadParams.cellDescription = this->cellDescription;

        __getTransactionEvent().waitForFinished();

        std::string fname = filename;
        if (isCheckpoint)
        {
            /* if checkpointFilename is relative, prepend with checkpointDirectory */
            if (!boost::filesystem::path(checkpointFilename).has_root_path())
            {
                fname = checkpointDirectory + std::string("/") + checkpointFilename;
            }
            else
            {
                fname = checkpointFilename;
            }

            mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);
        }
        else
        {
            mThreadParams.window = MovingWindow::getInstance().getWindow(currentStep);
        }

        for (uint32_t i = 0; i < simDim; ++i)
        {
            mThreadParams.localWindowToDomainOffset[i] = 0;
            if (mThreadParams.window.globalDimensions.offset[i] > localDomain.offset[i])
            {
                mThreadParams.localWindowToDomainOffset[i] =
                    mThreadParams.window.globalDimensions.offset[i] -
                    localDomain.offset[i];
            }
        }

        openH5File(fname);

        writeHDF5((void*) &mThreadParams);

        closeH5File();
    }

    void pluginLoad()
    {
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        /* It is important that we never change the mpi_pos after this point
         * because we get problems with the restart.
         * Otherwise we do not know which gpu must load the ghost parts around
         * the sliding window.
         */
        mpi_pos = gc.getPosition();
        mpi_size = gc.getGpuNodes();

        splashMpiPos.set(0, 0, 0);
        splashMpiSize.set(1, 1, 1);

        for (uint32_t i = 0; i < simDim; ++i)
        {
            splashMpiPos[i] = mpi_pos[i];
            splashMpiSize[i] = mpi_size[i];
        }


        /* only register for notify callback when .period is set on command line */
        if (notifyPeriod > 0)
        {
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
        }

        if (restartFilename == "")
        {
            restartFilename = checkpointFilename;
        }

        loaded = true;
    }

    void pluginUnload()
    {
        if (mThreadParams.dataCollector)
            mThreadParams.dataCollector->finalize();

        __delete(mThreadParams.dataCollector);
    }

    typedef PICToSplash<float_X>::type SplashFloatXType;

    static void writeMetaAttributes(ThreadParams *threadParams)
    {
        ColTypeUInt32 ctUInt32;
        ColTypeUInt64 ctUInt64;
        ColTypeDouble ctDouble;
        SplashFloatXType splashFloatXType;

        ParallelDomainCollector *dc = threadParams->dataCollector;
        uint32_t currentStep = threadParams->currentStep;

        /* write number of slides */
        const uint32_t slides = MovingWindow::getInstance().getSlideCounter(threadParams->currentStep);

        dc->writeAttribute(threadParams->currentStep,
                           ctUInt32, NULL, "sim_slides", &slides);

        /* write normed grid parameters */
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "delta_t", &DELTA_T);
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "cell_width", &CELL_WIDTH);
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "cell_height", &CELL_HEIGHT);
        if (simDim == DIM3)
        {
            dc->writeAttribute(currentStep, splashFloatXType, NULL, "cell_depth", &CELL_DEPTH);
        }

        /* write base units */
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_energy", &UNIT_ENERGY);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_length", &UNIT_LENGTH);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_speed", &UNIT_SPEED);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_time", &UNIT_TIME);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_mass", &UNIT_MASS);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_charge", &UNIT_CHARGE);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_efield", &UNIT_EFIELD);
        dc->writeAttribute(currentStep, ctDouble, NULL, "unit_bfield", &UNIT_BFIELD);

        /* write physical constants */
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "mue0", &MUE0);
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "eps0", &EPS0);
    }

    static void *writeHDF5(void *p_args)
    {
        ThreadParams *threadParams = (ThreadParams*) (p_args);
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        writeMetaAttributes(threadParams);

        /* y direction can be negative for first gpu*/
        DataSpace<simDim> particleOffset(localDomain.offset);
        particleOffset.y() -= threadParams->window.globalDimensions.offset.y();

        /* write all fields */
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing fields.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointFields, WriteFields<bmpl::_1> > forEachWriteFields;
            forEachWriteFields(threadParams);
        }
        else
        {
            ForEach<FileOutputFields, WriteFields<bmpl::_1> > forEachWriteFields;
            forEachWriteFields(threadParams);
        }
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing fields.");

        /* write all particle species */
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing particle species.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointParticles, WriteSpecies<bmpl::_1> > writeSpecies;
            writeSpecies(threadParams, std::string(), particleOffset);
        }
        else
        {
            ForEach<FileOutputParticles, WriteSpecies<bmpl::_1> > writeSpecies;
            writeSpecies(threadParams, std::string(), particleOffset);
        }
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing particle species.");

        PMACC_AUTO(idProviderState, IdProvider<simDim>::getState());
        log<picLog::INPUT_OUTPUT>("HDF5: Writing IdProvider state (NextId: %1%, maxNumProc: %2%)")
                % idProviderState.get<0>() % idProviderState.get<1>();
        WriteNDScalars<uint64_t, uint64_t>()(*threadParams,
                "picongpu/idProviderState", idProviderState.get<0>(),
                "maxNumProc", idProviderState.get<1>());

        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyPeriod;
    int64_t lastCheckpoint;
    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;
    std::string checkpointDirectory;

    uint32_t restartChunkSize;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;

    Dimensions splashMpiPos;
    Dimensions splashMpiSize;
};

} //namespace hdf5
} //namespace picongpu
