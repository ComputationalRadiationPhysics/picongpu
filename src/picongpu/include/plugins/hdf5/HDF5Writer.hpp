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
#include "plugins/hdf5/HDF5Writer.def"

#include "particles/frame_types.hpp"

#include <splash/splash.h>

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"
#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/particleToGrid/energyDensity.kernel"
#include "particles/operations/CountParticles.hpp"

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/GridLayout.hpp"
#include "dataManagement/ISimulationIO.hpp"
#include "moduleSystem/ModuleConnector.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "dimensions/TVec.h"

#include "plugins/IPluginModule.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>


#include "plugins/hdf5/WriteSpecies.hpp"


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
 * Implements the ISimulationIO interface.
 *
 * @param ElectronsBuffer class description for electrons
 * @param IonsBuffer class description for ions
 * @param simDim dimension of the simulation (2-3)
 */
class HDF5Writer : public ISimulationIO, public IPluginModule
{
public:

    /* filter particles by global position*/
    typedef bmpl::vector< typename GetPositionFilter<simDim>::type > usedFilters;
    typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;

private:

    /* filter is a rule which describe which particles should copy to host*/
    MyParticleFilter filter;

    template<typename UnitType>
    static std::vector<double> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<double> tmp(numComponents);
        for (uint i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }

    /** Write calculated fields to HDF5 file.
     *
     */
    template< typename T >
    struct GetDCFields
    {
    private:
        typedef typename T::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;

        static std::vector<double> getUnit()
        {
            typedef typename T::UnitValueType UnitType;
            UnitType unit = T::getUnit();
            return createUnit(unit, T::numComponents);
        }

    public:

        HDINLINE void operator()(RefWrapper<ThreadParams*> params, const DomainInformation domInfo)
        {
#ifndef __CUDA_ARCH__
            SplashType splashType;
            DataConnector &dc = DataConnector::getInstance();

            T* field = &(dc.getData<T > (T::getCommTag()));
            params.get()->gridLayout = field->getGridLayout();

            writeField(params.get(),
                       domInfo,
                       splashType,
                       GetNComponents<ValueType>::value,
                       T::getName(),
                       getUnit(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getCommTag());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     * and write them to hdf5.
     *
     * FieldTmp is calculated on device and than dumped to HDF5.
     */
    template< typename ThisSolver, typename ThisSpecies >
    struct GetDCFields<FieldTmpOperation<ThisSolver, ThisSpecies> >
    {

        /*
         * This is only a wrapper function to allow disable nvcc warnings.
         * Warning: calling a __host__ function from __host__ __device__
         * function.
         * Use of PMACC_NO_NVCC_HDWARNING is not possible if we call a virtual
         * method inside of the method were we disable the warnings.
         * Therefore we create this method and call a new method were we can
         * call virtual functions.
         */
        PMACC_NO_NVCC_HDWARNING
        HDINLINE void operator()(RefWrapper<ThreadParams*> tparam, const DomainInformation domInfo)
        {
            this->operator_impl(tparam, domInfo);
        }
    private:
        typedef typename FieldTmp::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;

        /** Create a name for the hdf5 identifier.
         */
        template< typename Solver, typename Species >
        static std::string getName()
        {
            std::stringstream str;
            str << FieldTmp::getName<Solver>();
            str << "_";
            str << Species::FrameType::getName();
            return str.str();
        }

        /** Get the unit for the result from the solver*/
        template<typename Solver>
        static std::vector<double> getUnit()
        {
            typedef typename FieldTmp::UnitValueType UnitType;
            UnitType unit = FieldTmp::getUnit<Solver>();
            const uint32_t components = GetNComponents<ValueType>::value;
            return createUnit(unit, components);
        }

        HINLINE void operator_impl(RefWrapper<ThreadParams*> params, const DomainInformation domInfo)
        {
            DataConnector &dc = DataConnector::getInstance();

            /*## update field ##*/

            /*load FieldTmp without copy data to host*/
            FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FIELD_TMP, true));
            /*load particle without copy particle data to host*/
            ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(
                                                                 ThisSpecies::FrameType::CommunicationTag, true));

            fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
            /*run algorithm*/
            fieldTmp->computeValue < CORE + BORDER, ThisSolver > (*speciesTmp, params.get()->currentStep);

            EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(fieldTmpEvent);
            /* copy data to host that we can write same to disk*/
            fieldTmp->getGridBuffer().deviceToHost();
            dc.releaseData(ThisSpecies::FrameType::CommunicationTag);
            /*## finish update field ##*/

            const uint32_t components = GetNComponents<ValueType>::value;
            SplashType splashType;

            params.get()->gridLayout = fieldTmp->getGridLayout();
            /*write data to HDF5 file*/
            writeField(params.get(),
                       domInfo,
                       splashType,
                       components,
                       getName<ThisSolver, ThisSpecies>(),
                       getUnit<ThisSolver>(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FIELD_TMP);

        }

    };

public:

    HDF5Writer() :
    filename("h5"),
    notifyFrequency(0)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~HDF5Writer()
    {

    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("hdf5.period", po::value<uint32_t > (&notifyFrequency)->default_value(0),
             "enable HDF5 IO [for each n-th step]")
            ("hdf5.file", po::value<std::string > (&filename)->default_value(filename),
             "HDF5 output file");
    }

    std::string moduleGetName() const
    {
        return "HDF5Writer";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {

        this->cellDescription = cellDescription;
    }

    __host__ void notify(uint32_t currentStep)
    {
        mThreadParams.currentStep = (int32_t) currentStep;
        mThreadParams.gridPosition = SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();
        mThreadParams.cellDescription = this->cellDescription;
        this->filter.setStatus(false);

        mThreadParams.window = MovingWindow::getInstance().getVirtualWindow(currentStep);

        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            //enable filters for sliding window and configurate position filter
            this->filter.setStatus(true);

            this->filter.setWindowPosition(mThreadParams.window.localOffset, mThreadParams.window.localSize);
        }

        __getTransactionEvent().waitForFinished();

        openH5File();

        writeHDF5((void*) &mThreadParams);

        closeH5File();

    }

private:

    void closeH5File()
    {
        if (mThreadParams.dataCollector != NULL)
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector with file: %1%") % filename;
            mThreadParams.dataCollector->close();
        }
    }

    void openH5File()
    {
        const uint32_t maxOpenFilesPerNode = 4;
        if ( mThreadParams.dataCollector == NULL)
        {
            GridController<simDim> &gc = GridController<simDim>::getInstance();
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
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % filename;
            mThreadParams.dataCollector->open(filename.c_str(), attr);
        }
        catch (DCException e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }

    }

    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            mThreadParams.gridPosition =
                SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();

            GridController<simDim> &gc = GridController<simDim>::getInstance();
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

            DataConnector::getInstance().registerObserver(this, notifyFrequency);
        }

        loaded = true;
    }

    void moduleUnload()
    {
        if (notifyFrequency > 0)
            __delete(mThreadParams.dataCollector);
    }

    static void writeField(ThreadParams *params, const DomainInformation domInfo,
                           CollectionType& colType,
                           const uint32_t nComponents, const std::string name,
                           std::vector<double> unit, void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("HDF5 write field: %1% %2% %3%") %
            name % nComponents % ptr;

        std::vector<std::string> name_lookup;
        {
            const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};
            for (uint32_t d = 0; d < nComponents; d++)
                name_lookup.push_back(name_lookup_tpl[d]);
        }

        /*data to describe source buffer*/
        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_full = field_layout.getDataSpace();
        DataSpace<simDim> field_no_guard = domInfo.domainSize;
        DataSpace<simDim> field_guard = field_layout.getGuard() + domInfo.localDomainOffset;
        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        globalSlideOffset.y()+=params->window.slides * params->window.localFullSize.y();

        Dimensions splashGlobalDomainOffset(0, 0, 0);
        Dimensions splashGlobalOffsetFile(0, 0, 0);
        Dimensions splashGlobalDomainSize(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashGlobalOffsetFile[d] = domInfo.domainOffset[d];
            splashGlobalDomainOffset[d] = domInfo.globalDomainOffset[d] + globalSlideOffset[d];
            splashGlobalDomainSize[d] = domInfo.globalDomainSize[d];
        }
        
        splashGlobalOffsetFile[1] = std::max(0, domInfo.domainOffset[1] -
                domInfo.globalDomainOffset[1]);


        for (uint32_t d = 0; d < nComponents; d++)
        {
            std::stringstream datasetName;
            datasetName << "fields/" << name;
            if (nComponents > 1)
                datasetName << "/" << name_lookup.at(d);
            
            Dimensions sizeSrcBuffer(1,1,1);
            Dimensions srcStride(nComponents, 1, 1);
            Dimensions sizeSrcData(1, 1,1);
            Dimensions srcOffset(0,0,0);

            for(uint32_t i=0;i<simDim;++i)
            {
                sizeSrcBuffer[i]=field_full[i];
                sizeSrcData[i]=field_no_guard[i];
                srcOffset[i]=field_guard[i];
            }
            sizeSrcBuffer[0]*=nComponents;
            srcOffset[0]*=nComponents;
            
            params->dataCollector->writeDomain(params->currentStep, /* id == time step */
                                               splashGlobalDomainSize,
                                               splashGlobalOffsetFile,
                                               colType, /* data type */
                                               simDim, /* NDims of the field data (scalar, vector, ...) */
                                               /* source buffer, stride, data size, offset */
                                               sizeSrcBuffer,
                                               srcStride,
                                               sizeSrcData,
                                               srcOffset,
                                               datasetName.str().c_str(), /* data set name */
                                               splashGlobalDomainOffset, /* \todo offset of the global domain */
                                               splashGlobalDomainSize, /* size of the global domain */
                                               DomainCollector::GridType,
                                               ptr);

            /*simulation attributes for data*/
            ColTypeDouble ctDouble;

            params->dataCollector->writeAttribute(params->currentStep,
                                                  ctDouble, datasetName.str().c_str(),
                                                  "sim_unit", &(unit.at(d)));
        }

    }

    static void *writeHDF5(void *p_args)
    {

        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        /* write number of slides to timestep in hdf5 file*/
        uint32_t slides = threadParams->window.slides;
        ColTypeUInt32 ctUInt32;
        threadParams->dataCollector->writeAttribute(threadParams->currentStep,
                                              ctUInt32, NULL, "sim_slides", &(slides));

        /* build clean domain info (picongpu view) */
        DomainInformation domInfo;
        /* set global offset (from physical origin) to our first gpu data area*/
        domInfo.localDomainOffset = threadParams->window.localOffset;
        domInfo.globalDomainOffset = threadParams->window.globalSimulationOffset;
        domInfo.globalDomainSize = threadParams->window.globalWindowSize;
        domInfo.domainOffset = threadParams->gridPosition;
        /* change only the offset of the first gpu
         * localDomainOffset is only non zero for the gpus on top
         */
        domInfo.domainOffset += domInfo.localDomainOffset;
        domInfo.domainSize = threadParams->window.localSize;

        /* y direction can be negative for first gpu*/
        DataSpace<simDim> particleOffset(threadParams->gridPosition);
        particleOffset.y() -= threadParams->window.globalSimulationOffset.y();

        /*print all fields*/
        ForEach<FileOutputFields, GetDCFields<void> > forEachGetFields;
        forEachGetFields(ref(threadParams), domInfo);

        /*print all particle species*/
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing particle species.");
        ForEach<FileOutputParticles, WriteSpecies<void> > writeSpecies;
        writeSpecies(ref(threadParams), std::string(), domInfo, particleOffset);
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing particle species.");


        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            /* data domain = domain inside the sliding window
             * ghost domain = domain under the data domain (is laying only on bottom gpus)
             * end of data domain is the beginning of the ghost domain
             */
            domInfo.globalDomainOffset.y() += domInfo.globalDomainSize.y();
            domInfo.domainOffset.y() = domInfo.globalDomainOffset.y();
            domInfo.domainSize = threadParams->window.localFullSize;
            domInfo.domainSize.y() -= threadParams->window.localSize.y();
            domInfo.globalDomainSize = threadParams->window.globalSimulationSize;
            domInfo.globalDomainSize.y() -= domInfo.globalDomainOffset.y();
            domInfo.localDomainOffset = DataSpace<simDim > ();
            /* only important for bottom gpus*/
            domInfo.localDomainOffset.y() = threadParams->window.localSize.y();

            particleOffset = threadParams->gridPosition;
            particleOffset.y() = -threadParams->window.localSize.y();

            if (threadParams->window.isBottom == false)
            {
                /* set size for all gpu to zero which are not bottom gpus*/
                domInfo.domainSize.y() = 0;
            }
            /* for restart we only need bottom ghosts for particles */
            log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing particle species bottom.");
            /* print all particle species */
            writeSpecies(ref(threadParams), std::string("_ghosts"), domInfo, particleOffset);
            log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing particle species bottom.");
        }
        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyFrequency;
    std::string filename;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;

    Dimensions splashMpiPos;
    Dimensions splashMpiSize;
};

} //namespace hdf5
} //namespace picongpu

