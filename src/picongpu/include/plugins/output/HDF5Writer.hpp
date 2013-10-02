/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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

#include <pthread.h>
#include <cassert>
#include <sstream>
#include <list>
#include <vector>

#include "types.h"
#include "simulation_types.hpp"
#include "particles/frame_types.hpp"

#include "DomainCollector.hpp"
#include "basetypes/ColTypeDim.hpp"
#include "basetypes/ColTypeFloat.hpp"
#include "basetypes/ColTypeDouble.hpp"
#include "basetypes/ColTypeInt.hpp"
#include "basetypes/ColTypeBool.hpp"
#include "basetypes/ColTypeInt3Array.hpp"
#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.def"
#include "fields/FieldTmp.hpp"
#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/particleToGrid/energyDensity.kernel"

#include "dataManagement/DataConnector.hpp"
#include "particles/memory/frames/FrameContainer.hpp"
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

namespace picongpu
{

using namespace PMacc;

using namespace DCollector;
namespace bmpl = boost::mpl;

namespace po = boost::program_options;

/**
 * Writes simulation data to hdf5 files.
 * Implements the ISimulationIO interface.
 *
 * @param ElectronsBuffer class description for electrons
 * @param IonsBuffer class description for ions
 * @param DIM dimension of the simulation (2-3)
 */
template<class ElectronsBuffer, class IonsBuffer, unsigned DIM>
class HDF5Writer : public ISimulationIO, public IPluginModule
{
private:
    typedef bmpl::vector< PositionFilter3D<> > usedFilters;
    typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;

#if (ENABLE_ELECTRONS == 1)
    typedef FrameContainer<typename ElectronsBuffer::BufferType, TVec < 2048, 16, 1 >,
    MappingDesc::SuperCellSize, MyParticleFilter> MyFrameContainerE;

    typedef typename MyFrameContainerE::ParticleType MyBigFrameE;
#endif
#if (ENABLE_IONS == 1)
    typedef FrameContainer<typename IonsBuffer::BufferType, TVec < 2048, 16, 1 >,
    MappingDesc::SuperCellSize, MyParticleFilter> MyFrameContainerI;

    typedef typename MyFrameContainerI::ParticleType MyBigFrameI;
#endif

    MyParticleFilter filter;

    struct ThreadParams
    {
        uint32_t currentStep;
        DCollector::DomainCollector *dataCollector;
        GridLayout<DIM> gridLayout;
        DataSpace<DIM> gridPosition;
#if (ENABLE_ELECTRONS == 1)
        MyFrameContainerE *frameContainerE;
#endif
#if (ENABLE_IONS == 1)
        MyFrameContainerI *frameContainerI;
#endif

        VirtualWindow window;
    };

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

        static std::vector<double> getUnit()
        {
            typedef typename T::UnitValueType UnitType;
            UnitType unit = T::getUnit();
            return createUnit(unit, T::numComponents);
        }

    public:

        HDINLINE void operator()(RefWrapper<ThreadParams*> params)
        {
#ifndef __CUDA_ARCH__
            DCollector::ColTypeFloat ctFloat;

            DataConnector &dc = DataConnector::getInstance();

            T* field = &(dc.getData<T > (T::getCommTag()));
            params.get()->gridLayout = field->getGridLayout();

            writeField(params.get(),
                       ctFloat,
                       T::numComponents,
                       T::getName(),
                       getUnit(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getCommTag());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     *  and write them to hdf5.
     *
     *  FieldTmp is calculated on device and than dumped to HDF5.
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
        HDINLINE void operator()(RefWrapper<ThreadParams*> tparam)
        {
            this->operator_impl(tparam);
        }
    private:
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
            return createUnit(unit, FieldTmp::numComponents);
        }

        HINLINE void operator_impl(RefWrapper<ThreadParams*> params)
        {

            DCollector::ColTypeFloat ctFloat;

            DataConnector &dc = DataConnector::getInstance();

            /*## update field ##*/
            
            /*load FieldTmp without copy data to host*/
            FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FIELD_TMP, true));
            /*load particle without copy particle data to host*/
            ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::CommunicationTag, true));

            fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
            /*run algorithm*/
            fieldTmp->computeValue < CORE + BORDER, ThisSolver > (*speciesTmp, params.get()->currentStep);

            EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(fieldTmpEvent);
            /* copy data to host that we can write same to disk*/
            fieldTmp->getGridBuffer().deviceToHost();
            dc.releaseData(ThisSpecies::FrameType::CommunicationTag);
            /*## finish update field ##*/

            params.get()->gridLayout =fieldTmp->getGridLayout();
            /*write data to HDF5 file*/
            writeField(params.get(),
                       ctFloat,
                       FieldTmp::numComponents,
                       getName<ThisSolver, ThisSpecies>(),
                       getUnit<ThisSolver>(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FIELD_TMP);

        }

    };

public:

    HDF5Writer() :
    filename("h5"),
    notifyFrequency(0),
    compression(false),
    continueFile(false)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~HDF5Writer()
    {

    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("hdf5.period", po::value<uint32_t > (&notifyFrequency)->default_value(0), "enable HDF5 IO  [for each n-th step]")
            ("hdf5.file", po::value<std::string > (&filename)->default_value(filename), "HDF5 output file")
            ("hdf5.compression", po::value<bool > (&compression)->zero_tokens(), "enable HDF5 compression")
            ("hdf5.continue", po::value<bool > (&continueFile)->zero_tokens(), "continue existing HDF5 file instead of creating a new one");
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
        DataConnector &dc = DataConnector::getInstance();

        mThreadParams.currentStep = currentStep;
        mThreadParams.gridPosition = SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();
        this->filter.setStatus(false);

        mThreadParams.window = MovingWindow::getInstance().getVirtualWindow(currentStep);

        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            //enable  filters for sliding window and configurate position filter
            this->filter.setStatus(true);

            this->filter.setWindowPosition(mThreadParams.window.localOffset, mThreadParams.window.localSize);
        }

#if (ENABLE_ELECTRONS == 1)
        ElectronsBuffer *electrons = &(dc.getData<ElectronsBuffer > (PAR_ELECTRONS));
        if (mThreadParams.frameContainerE != NULL)
        {
            delete mThreadParams.frameContainerE;
            mThreadParams.frameContainerE = NULL;
        }
        mThreadParams.frameContainerE = new MyFrameContainerE(electrons->getParticlesBuffer(),
                                                              mThreadParams.gridLayout.getGuard() / MappingDesc::SuperCellSize::getDataSpace(), this->filter);
#endif 
#if (ENABLE_IONS == 1)
        IonsBuffer *ions = &(dc.getData<IonsBuffer > (PAR_IONS));
        if (mThreadParams.frameContainerI != NULL)
        {
            delete mThreadParams.frameContainerI;
            mThreadParams.frameContainerI = NULL;
        }
        mThreadParams.frameContainerI = new MyFrameContainerI(ions->getParticlesBuffer(),
                                                              mThreadParams.gridLayout.getGuard() / MappingDesc::SuperCellSize::getDataSpace(), this->filter);
#endif


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
            mThreadParams.dataCollector->close();
            delete mThreadParams.dataCollector;
            mThreadParams.dataCollector = NULL;
        }
    }

    void openH5File()
    {
        const uint32_t maxOpenFilesPerNode = 4;
        mThreadParams.dataCollector = new DCollector::DomainCollector(maxOpenFilesPerNode);

        // set attributes for datacollector files
        DCollector::DataCollector::FileCreationAttr attr;
        attr.enableCompression = this->compression;

        if (continueFile)
            attr.fileAccType = DCollector::DataCollector::FAT_WRITE;
        else
            attr.fileAccType = DCollector::DataCollector::FAT_CREATE;


        attr.mpiPosition.set(0, 0, 0);
        attr.mpiSize.set(1, 1, 1);

        for (uint32_t i = 0; i < DIM; ++i)
        {
            attr.mpiPosition[i] = mpi_pos[i];
            attr.mpiSize[i] = mpi_size[i];
        }

        // open datacollector
        try
        {
            mThreadParams.dataCollector->open(filename.c_str(), attr);
        }
        catch (DCollector::DCException e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open datacollector");
        }

        DCollector::Dimensions global_offset(10, 11, 12);
        DCollector::ColTypeDim ctDim;
        mThreadParams.dataCollector->writeGlobalAttribute(ctDim, "global_offset", &global_offset);

        continueFile = true; //set continue for the next open

    }

    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            mThreadParams.gridPosition = SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset();
#if (ENABLE_ELECTRONS == 1)
            mThreadParams.frameContainerE = NULL;
#endif
#if (ENABLE_IONS == 1)
            mThreadParams.frameContainerI = NULL;
#endif
            GridController<DIM> &gc = GridController<DIM>::getInstance();
            mpi_pos = gc.getPosition();
            mpi_size = gc.getGpuNodes();

            DataConnector::getInstance().registerObserver(this, notifyFrequency);
        }

        loaded = true;
    }

    void moduleUnload()
    {
        if (notifyFrequency > 0)
        {


            // cleanup
#if (ENABLE_ELECTRONS == 1)
            if (mThreadParams.frameContainerE != NULL)
            {
                delete mThreadParams.frameContainerE;
                mThreadParams.frameContainerE = NULL;
            }
#endif
#if (ENABLE_IONS == 1)
            if (mThreadParams.frameContainerI != NULL)
            {
                delete mThreadParams.frameContainerI;
                mThreadParams.frameContainerI = NULL;
            }
#endif

        }
    }

    static void writeField(ThreadParams *params, DCollector::CollectionType& colType,
                           const uint32_t dims, const std::string name, std::vector<double> unit, void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("HDF5 write field: %1% %2% %3%") %
            name % dims % ptr;

        std::vector<std::string> name_lookup;
        {
            const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};
            for (uint32_t d = 0; d < dims; d++)
                name_lookup.push_back(name_lookup_tpl[d]);
        }

        GridLayout<DIM> field_layout = params->gridLayout;
        DataSpace<DIM> field_full = field_layout.getDataSpace();
        DataSpace<DIM> field_no_guard = params->window.localSize; //field_layout.getDataSpaceWithoutGuarding();
        DataSpace<DIM> field_guard = field_layout.getGuard() + params->window.localOffset;

        DataSpace<DIM> sim_offset = params->gridPosition - params->window.globalSimulationOffset;

        /*simulation attributes for data*/
        DCollector::ColTypeDouble ctDouble;

        DCollector::Dimensions domain_offset(0, 0, 0);
        DCollector::Dimensions domain_size(1, 1, 1);

        ///\todo these might be deprecated !
        DCollector::Dimensions sim_size(0, 0, 0);
        DCollector::Dimensions sim_global_offset(0, 0, 0);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            sim_size[d] = field_no_guard[d];
            /*fields of first gpu in simulation are NULL point*/
            if (sim_offset[d] > 0)
            {
                sim_global_offset[d] = sim_offset[d];
                domain_offset[d] = sim_offset[d];
            }

            domain_size[d] = field_no_guard[d];
        }



        //only write data if we have data
        // if (field_no_guard.y() > 0)
        {
            for (uint32_t d = 0; d < dims; d++)
            {
                std::stringstream str;
                str << name;
                if (dims > 1)
                    str << "_" << name_lookup.at(d);

                params->dataCollector->writeDomain(params->currentStep, colType, DIM,
                                                   DCollector::Dimensions(field_full[0] * dims, field_full[1], field_full[2]),
                                                   DCollector::Dimensions(dims, 1, 1),
                                                   DCollector::Dimensions(field_no_guard[0], field_no_guard[1], field_no_guard[2]),
                                                   DCollector::Dimensions(field_guard[0] * dims + d, field_guard[1], field_guard[2]),
                                                   str.str().c_str(),
                                                   domain_offset,
                                                   domain_size,
                                                   DomainCollector::GridType,
                                                   ptr);

                params->dataCollector->writeAttribute(params->currentStep, DCollector::ColTypeDim(), str.str().c_str(), "sim_size", &sim_size);
                params->dataCollector->writeAttribute(params->currentStep, DCollector::ColTypeDim(), str.str().c_str(), "sim_global_offset", &sim_global_offset);
                params->dataCollector->writeAttribute(params->currentStep, ctDouble, str.str().c_str(), "sim_unit", &(unit.at(d)));
            }
        }
    }

    static void writeParticlesIntern(ThreadParams *params, DataSpace<DIM>& sim_offset, DataSpace<DIM3>& sim_size, DCollector::CollectionType& colType,
                                     const uint32_t dims, const uint32_t elements, const char *prefix,
                                     const char *name, const std::string name_lookup[], double* unit, void *ptr)
    {
        DCollector::ColTypeDouble ctDouble;
        DataSpace<DIM> field_no_guard = sim_size;

        DCollector::Dimensions domain_offset(0, 0, 0);
        DCollector::Dimensions domain_size(1, 1, 1);

        ///\todo this might be deprecated
        DCollector::Dimensions sim_global_offset(0, 0, 0);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            if (sim_offset[d] > 0)
            {
                sim_global_offset[d] = sim_offset[d];
                domain_offset[d] = sim_offset[d];
            }
            domain_size[d] = field_no_guard[d];
        }

        for (uint32_t d = 0; d < dims; d++)
        {
            std::stringstream str;
            str << prefix << name;
            if (name_lookup != NULL)
                str << "_" << name_lookup[d];

            params->dataCollector->appendDomain(params->currentStep,
                                                colType,
                                                elements,
                                                d,
                                                dims,
                                                str.str().c_str(),
                                                domain_offset,
                                                domain_size,
                                                ptr);

            if (unit != NULL)
                params->dataCollector->writeAttribute(params->currentStep, ctDouble, str.str().c_str(), "sim_unit", &(unit[d]));
            params->dataCollector->writeAttribute(params->currentStep, DCollector::ColTypeDim(), str.str().c_str(), "sim_global_offset", &sim_global_offset);
        }
    }

    template <class FrameContainerType, class BigFrameType>
    static void writeParticles(ThreadParams *params, FrameContainerType *frameContainer,
                               DataSpace<DIM>& sim_offset, DataSpace<DIM3>& sim_size, DataSpace<DIM3> pysicalToLogicalOffset,
                               std::string prefix)
    {
        // Keep iterating over frameContainer to get new big frames 
        // which should be written to hdf5 using the dataCollector.
        bool hasNext = false;

        DCollector::ColTypeFloat ctFloat;
        DCollector::ColTypeDouble ctDouble;
        DCollector::ColTypeInt ctInt;
#if(ENABLE_RADIATION == 1) &&((RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0))
        DCollector::ColTypeBool ctBool;
#endif

#if (SIMDIM == DIM2)
        const std::string name_lookup[] = {"x", "y"};
#elif (SIMDIM == DIM3)
        const std::string name_lookup[] = {"x", "y", "z"};
#endif

        size_t particleCounter = 0;


        double unitMomentum[] = {UNIT_ENERGY, UNIT_ENERGY, UNIT_ENERGY};
        double unitPos[] = {SI::CELL_WIDTH_SI, SI::CELL_HEIGHT_SI, SI::CELL_DEPTH_SI};
        double unitWeighting[] = {1.};


        // loop over all big frames
        // note: Write calls have to be performed even if the number 
        // of elements is zero to allocate the datasets in the file.
        do
        {
            size_t elements = 0;
            BigFrameType frame;
            if (frameContainer != NULL)
            {
                frame = frameContainer->getNextBigFrame(hasNext);
                elements = frameContainer->getElemCount();
            }
            particleCounter += elements;

            // write position of particle in cell
            writeParticlesIntern(params,
                                 sim_offset, sim_size,
                                 ctFloat,
                                 DIM,
                                 elements,
                                 prefix.c_str(),
                                 "_relative_position",
                                 name_lookup,
                                 unitPos,
                                 frame.getPosition().getPointer());

            // update gpu-relative cell positions to simulation-relative positions
            for (size_t i = 0; i < elements; ++i)
                for (size_t d = 0; d < DIM; ++d)
                    frame.getGlobalCellIdx()[i][d] += pysicalToLogicalOffset[d];

            // write simulation-relative position of cell
            writeParticlesIntern(params,
                                 sim_offset, sim_size,
                                 ctInt,
                                 DIM,
                                 elements,
                                 prefix.c_str(),
                                 "_global_cell_pos",
                                 name_lookup,
                                 unitPos,
                                 frame.getGlobalCellIdx().getPointer());

            // write momentum of particle
            writeParticlesIntern(params,
                                 sim_offset, sim_size,
                                 ctFloat,
                                 DIM,
                                 elements,
                                 prefix.c_str(),
                                 "_momentum",
                                 name_lookup,
                                 unitMomentum,
                                 frame.getMomentum().getPointer());

            // write weighting of particle
            writeParticlesIntern(params,
                                 sim_offset, sim_size,
                                 ctFloat,
                                 1,
                                 elements,
                                 prefix.c_str(),
                                 "_weighting",
                                 NULL,
                                 unitWeighting,
                                 frame.getWeighting().getPointer());
#if(ENABLE_RADIATION == 1)
            // write old memonetum
            writeParticlesIntern(params,
                                 sim_offset, sim_size,
                                 ctFloat,
                                 DIM,
                                 elements,
                                 prefix.c_str(),
                                 "_momentum_mt1",
                                 name_lookup,
                                 unitMomentum,
                                 frame.getMomentum_mt1().getPointer());
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
            writeParticlesIntern(params,
                                 sim_offset, sim_size,
                                 ctBool,
                                 1,
                                 elements,
                                 prefix.c_str(),
                                 "_radiationFlag",
                                 NULL,
                                 NULL,
                                 frame.getRadiationFlag().getPointer());
#endif
#endif
        }
        while (hasNext && frameContainer != NULL);

        params->dataCollector->writeAttribute(params->currentStep, ctDouble, (prefix + std::string("_weighting")).c_str(), "sim_unit", unitWeighting);

    }

    static void *writeHDF5(void *p_args)
    {

        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        /*print all fields*/
        ForEach<Hdf5OutputFields, GetDCFields<void> > forEachGetFields;
        forEachGetFields(ref(threadParams));

        // write fields
        /// \todo this should be a type trait
        DCollector::ColTypeFloat ctFloat;

        // write particles
        DataSpace<DIM> sim_offset = threadParams->gridPosition - threadParams->window.globalSimulationOffset;
        DataSpace<DIM> localOffset = threadParams->window.localOffset;
        DataSpace<DIM> localSize = threadParams->window.localSize;

#if (ENABLE_ELECTRONS == 1)
        threadParams->frameContainerE->getFilter().setWindowPosition(localOffset, localSize);
        writeParticles<MyFrameContainerE, MyBigFrameE > (threadParams,
                                                         threadParams->frameContainerE,
                                                         sim_offset, localSize, sim_offset,
                                                         ElectronsBuffer::FrameType::getName());
#endif
#if (ENABLE_IONS == 1)
        threadParams->frameContainerI->getFilter().setWindowPosition(localOffset, localSize);
        writeParticles<MyFrameContainerI, MyBigFrameI > (threadParams,
                                                         threadParams->frameContainerI,
                                                         sim_offset, localSize, sim_offset,
                                                         IonsBuffer::FrameType::getName());
#endif
        if (MovingWindow::getInstance().isSlidingWindowActive())
        {

            DataSpace<DIM> sim_offset = threadParams->gridPosition;

            DataSpace<DIM> localOffset;

            DataSpace<DIM> localSize = threadParams->window.localSize;
            localSize.y() = threadParams->window.globalSimulationOffset.y();


            sim_offset = threadParams->gridPosition;

#if (ENABLE_ELECTRONS == 1)
            PMACC_AUTO(container, threadParams->frameContainerE);
            if (!threadParams->window.isTop)
                container = NULL;
            threadParams->frameContainerE->clear();
            threadParams->frameContainerE->getFilter().setWindowPosition(localOffset, localSize);
            std::stringstream strNameTopE;
            strNameTopE << "_top_";
            strNameTopE << ElectronsBuffer::FrameType::getName();
            writeParticles<MyFrameContainerE, MyBigFrameE > (threadParams,
                                                             container,
                                                             sim_offset, localSize, DataSpace<DIM > (),
                                                             strNameTopE.str());
#endif
#if (ENABLE_IONS == 1)
            PMACC_AUTO(containerI, threadParams->frameContainerI);
            if (!threadParams->window.isTop)
                containerI = NULL;
            threadParams->frameContainerI->clear();
            threadParams->frameContainerI->getFilter().setWindowPosition(localOffset, localSize);
            std::stringstream strNameTopI;
            strNameTopI << "_top_";
            strNameTopI << IonsBuffer::FrameType::getName();
            writeParticles<MyFrameContainerI, MyBigFrameI > (threadParams,
                                                             containerI,
                                                             sim_offset, localSize, DataSpace<DIM > (),
                                                             strNameTopI.str());
#endif

            sim_offset = threadParams->gridPosition;
            sim_offset.y() += threadParams->window.localSize.y();

            localOffset = DataSpace<DIM3 > ();
            localOffset.y() = threadParams->window.localSize.y();

            localSize = threadParams->window.localFullSize;
            localSize.y() -= threadParams->window.localSize.y();



#if (ENABLE_ELECTRONS == 1)
            PMACC_AUTO(containerBottom, threadParams->frameContainerE);
            if (!threadParams->window.isBottom)
                containerBottom = NULL;
            threadParams->frameContainerE->clear();
            threadParams->frameContainerE->getFilter().setWindowPosition(localOffset,
                                                                         localSize);
            std::stringstream strNameBottomE;
            strNameBottomE << "_bottom_";
            strNameBottomE << ElectronsBuffer::FrameType::getName();
            writeParticles<MyFrameContainerE, MyBigFrameE > (threadParams,
                                                             containerBottom,
                                                             sim_offset, localSize, DataSpace<DIM > (),
                                                             strNameBottomE.str());
#endif
#if (ENABLE_IONS == 1)
            PMACC_AUTO(containerBottomI, threadParams->frameContainerI);
            if (!threadParams->window.isBottom)
                containerBottomI = NULL;
            threadParams->frameContainerI->clear();
            threadParams->frameContainerI->getFilter().setWindowPosition(localOffset,
                                                                         localSize);
            std::stringstream strNameBottomI;
            strNameBottomI << "_bottom_";
            strNameBottomI << IonsBuffer::FrameType::getName();
            writeParticles<MyFrameContainerI, MyBigFrameI > (threadParams,
                                                             containerBottomI,
                                                             sim_offset, localSize, DataSpace<DIM > (),
                                                             strNameBottomI.str());
#endif
        }




        DataConnector &dc = DataConnector::getInstance();

        // release particle data
#if (ENABLE_ELECTRONS == 1)
        dc.releaseData(PAR_ELECTRONS);
#endif
#if (ENABLE_IONS == 1)
        dc.releaseData(PAR_IONS);
#endif

        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyFrequency;
    std::string filename;
    bool compression;
    bool continueFile;

    DataSpace<DIM> mpi_pos;
    DataSpace<DIM> mpi_size;

};

} //namepsace picongpu

