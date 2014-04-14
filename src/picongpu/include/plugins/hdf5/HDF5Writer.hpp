/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
#include <cassert>
#include <sstream>
#include <list>
#include <vector>

#include "types.h"
#include "simulation_types.hpp"
#include "simulation_defines.hpp"
#include "plugins/hdf5/HDF5Writer.def"

#include "particles/frame_types.hpp"

#include <splash/splash.h>

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"
#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/operations/CountParticles.hpp"

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/GridLayout.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "dimensions/TVec.h"

#include "plugins/ISimulationPlugin.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>

#include "plugins/hdf5/WriteFields.hpp"
#include "plugins/hdf5/WriteSpecies.hpp"
#include "plugins/hdf5/restart/RestartFieldLoader.hpp"
#include "plugins/hdf5/restart/RestartParticleLoader.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"

namespace picongpu
{

namespace hdf5
{

using namespace PMacc;

using namespace splash;
namespace bmpl = boost::mpl;

namespace po = boost::program_options;

/**
 * Writes simulation data to hdf5 files using libSplash.
 * Implements the ISimulationPlugin interface.
 */
class HDF5Writer : public ISimulationPlugin
{
public:

    HDF5Writer() :
    filename("h5"),
    checkpointFilename(""),
    restartFilename(""),
    notifyFrequency(0),
    lastCheckpoint(-1)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~HDF5Writer()
    {

    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("hdf5.period", po::value<uint32_t > (&notifyFrequency)->default_value(0),
             "enable HDF5 IO [for each n-th step]")
            ("hdf5.file", po::value<std::string > (&filename)->default_value(filename),
             "HDF5 output filename (prefix)")
            ("hdf5.checkpoint-file", po::value<std::string > (&checkpointFilename),
             "Optional HDF5 checkpoint filename (prefix)")
            ("hdf5.restart-file", po::value<std::string > (&restartFilename),
             "HDF5 restart filename (prefix)");
    }

    std::string pluginGetName() const
    {
        return "HDF5Writer";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {

        this->cellDescription = cellDescription;
    }

    __host__ void notify(uint32_t currentStep)
    {
        if ((int64_t)currentStep > lastCheckpoint)
            notificationReceived(currentStep, false);
    }
    
    void checkpoint(uint32_t currentStep)
    {
        notificationReceived(currentStep, true);
        lastCheckpoint = currentStep;
    }
    
    void restart(uint32_t restartStep)
    {
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
        
        /* open datacollector */
        try
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % restartFilename;
            mThreadParams.dataCollector->open(restartFilename.c_str(), attr);
        }
        catch (DCException e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }
        
        /* load number of slides to initialize MovingWindow */
        int slides = 0;
        mThreadParams.dataCollector->readAttribute(restartStep, NULL, "sim_slides", &slides);

        /* apply slides to set gpus to last/written configuration */
        log<picLog::INPUT_OUTPUT > ("Setting slide count for moving window to %1%") % slides;
        MovingWindow::getInstance().setSlideCounter((uint32_t) slides);
        gc.setNumSlides(slides);
        
        DataSpace<simDim> gridPosition = 
                Environment<simDim>::get().SubGrid().getSimulationBox().getGlobalOffset();
        log<picLog::INPUT_OUTPUT > ("Grid position is %1%") % gridPosition.toString();
        
        ThreadParams *params = &mThreadParams;
        
        /* load all fields */
        ForEach<FileRestartFields, LoadFields<void> > forEachLoadFields;
        forEachLoadFields(ref(params));
        
        /* load all particles */
        ForEach<FileRestartParticles, LoadParticles<void> > forEachLoadSpecies;
        forEachLoadSpecies(ref(params), gridPosition);
        
        /* close datacollector */
        log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector with file: %1%") % restartFilename;
        mThreadParams.dataCollector->close();
        __delete(mThreadParams.dataCollector);
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
        catch (DCException e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }

    }
    
    void notificationReceived(uint32_t currentStep, bool isCheckpoint)
    {
        mThreadParams.isCheckpoint = isCheckpoint;
        mThreadParams.currentStep = (int32_t) currentStep;
        mThreadParams.gridPosition = Environment<simDim>::get().SubGrid().getSimulationBox().getGlobalOffset();
        mThreadParams.cellDescription = this->cellDescription;
        mThreadParams.window = MovingWindow::getInstance().getVirtualWindow(currentStep);

        __getTransactionEvent().waitForFinished();

        std::string fname = filename;
        if (isCheckpoint && (checkpointFilename != ""))
            fname = checkpointFilename;
        
        openH5File(fname);

        writeHDF5((void*) &mThreadParams);

        closeH5File();
    }

    void pluginLoad()
    {
        if (notifyFrequency > 0)
        {
            mThreadParams.gridPosition =
                Environment<simDim>::get().SubGrid().getSimulationBox().getGlobalOffset();

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

            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
        }
        
        if (restartFilename == "")
        {
            restartFilename = filename;
        }

        loaded = true;
    }

    void pluginUnload()
    {
        if (notifyFrequency > 0)
            __delete(mThreadParams.dataCollector);
    }

    typedef PICToSplash<float_X>::type SplashFloatXType;

    static void writeMetaAttributes(ThreadParams *threadParams)
    {
        ColTypeUInt32 ctUInt32;
        ColTypeDouble ctDouble;
        SplashFloatXType splashFloatXType;

        ParallelDomainCollector *dc = threadParams->dataCollector;
        uint32_t currentStep = threadParams->currentStep;

        /* write number of slides */
        uint32_t slides = threadParams->window.slides;

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

        /* write physical constants */
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "mue0", &MUE0);
        dc->writeAttribute(currentStep, splashFloatXType, NULL, "eps0", &EPS0);
    }

    static void *writeHDF5(void *p_args)
    {
        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        writeMetaAttributes(threadParams);

        /* get clean domain info (picongpu view) */
        DomainInformation domInfo = 
                MovingWindow::getInstance().getActiveDomain(threadParams->currentStep);

        /* y direction can be negative for first gpu*/
        DataSpace<simDim> particleOffset(threadParams->gridPosition);
        particleOffset.y() -= threadParams->window.globalSimulationOffset.y();

        /*print all fields*/
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing fields.");
        ForEach<FileOutputFields, WriteFields<void> > forEachWriteFields;
        forEachWriteFields(ref(threadParams), domInfo);
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing fields.");

        /*print all particle species*/
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing particle species.");
        ForEach<FileOutputParticles, WriteSpecies<void> > writeSpecies;
        writeSpecies(ref(threadParams), std::string(), domInfo, particleOffset);
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing particle species.");


        if (threadParams->isCheckpoint && MovingWindow::getInstance().isSlidingWindowActive())
        {
            DomainInformation domInfoGhosts = 
                    MovingWindow::getInstance().getGhostDomain(threadParams->currentStep);

            particleOffset = threadParams->gridPosition;
            particleOffset.y() = -threadParams->window.localSize.y();

            /* for restart we only need bottom ghosts for particles */
            log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing particle species bottom.");
            /* print all particle species */
            writeSpecies(ref(threadParams), std::string("_ghosts"), domInfoGhosts, particleOffset);
            log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing particle species bottom.");
        }
        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyFrequency;
    int64_t lastCheckpoint;
    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;

    Dimensions splashMpiPos;
    Dimensions splashMpiSize;
};

} //namespace hdf5
} //namespace picongpu

