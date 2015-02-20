/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "simulation_defines.hpp"
#include "types.h"

#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "memory/buffers/GridBuffer.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"

#include "dataManagement/DataConnector.hpp"
#include "math/Vector.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/SharedBox.hpp"
#include "memory/boxes/PitchedBox.hpp"
#include "memory/buffers/GridBuffer.hpp"

#include "simulationControl/MovingWindow.hpp"

#include "mappings/kernel/MappingDescription.hpp"
//c includes
#include "sys/stat.h"
#include "mappings/simulation/GridController.hpp"

#include <string>
#include "memory/boxes/PitchedBox.hpp"
#include "plugins/output/header/MessageHeader.hpp"
#include "plugins/output/GatherSlice.hpp"
#include "plugins/ILightweightPlugin.hpp"

namespace picongpu
{
using namespace PMacc;

template<class ParBox, class Mapping, typename Type_>
__global__ void
kernelParticleDensity(ParBox pb,
                      DataBox<PitchedBox<Type_, DIM2> > image,
                      DataSpace<DIM2> transpose,
                      int slice,
                      uint32_t globalOffset,
                      uint32_t sliceDim,
                      Mapping mapper)
{
    typedef typename ParBox::FrameType FRAME;
    typedef typename MappingDesc::SuperCellSize Block;
    __shared__ FRAME *frame;
    __shared__ bool isValid;
    __syncthreads(); /*wait that all shared memory is initialised*/

    bool isImageThread = false;

    const DataSpace<simDim> threadId(threadIdx);
    const DataSpace<DIM2> localCell(threadId[transpose.x()], threadId[transpose.y()]);
    const DataSpace<simDim> block = mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx));
    const DataSpace<simDim> blockOffset((block - 1) * Block::toRT());


    int localId = threadIdx.z * Block::x::value * Block::y::value + threadIdx.y * Block::x::value + threadIdx.x;


    if (localId == 0)
        isValid = false;
    __syncthreads();

    //\todo: guard size should not be set to (fixed) 1 here
    const DataSpace<simDim> realCell(blockOffset + threadId); //delete guard from cell idx

#if(SIMDIM==DIM3)
    uint32_t globalCell = realCell[sliceDim] + globalOffset;

    if (globalCell == slice)
#endif
    {
        isValid = true;
        isImageThread = true;
    }
    __syncthreads();

    if (!isValid)
        return;

    /*index in image*/
    DataSpace<DIM2> imageCell(
                              realCell[transpose.x()],
                              realCell[transpose.y()]);


    // counter is always DIM2
    typedef DataBox < PitchedBox< float_X, DIM2 > > SharedMem;
    extern __shared__ float_X shBlock[];
    __syncthreads(); /*wait that all shared memory is initialised*/

    const DataSpace<simDim> blockSize(blockDim);
    SharedMem counter(PitchedBox<float_X, DIM2 > ((float_X*) shBlock,
                                                  DataSpace<DIM2 > (),
                                                  blockSize[transpose.x()] * sizeof (float_X)));

    if (isImageThread)
    {
        counter(localCell) = float_X(0.0);
    }


    if (localId == 0)
    {
        frame = &(pb.getFirstFrame(block, isValid));
    }
    __syncthreads();

    while (isValid) //move over all Frames
    {
        PMACC_AUTO(particle, (*frame)[localId]);
        if (particle[multiMask_] == 1)
        {
            int cellIdx = particle[localCellIdx_];
            // we only draw the first slice of cells in the super cell (z == 0)
            const DataSpace<simDim> particleCellId(DataSpaceOperations<simDim>::template map<Block > (cellIdx));
#if(SIMDIM==DIM3)
            uint32_t globalParticleCell = particleCellId[sliceDim] + globalOffset + blockOffset[sliceDim];
            if (globalParticleCell == slice)
#endif
            {
                const DataSpace<DIM2> reducedCell(particleCellId[transpose.x()], particleCellId[transpose.y()]);
                atomicAddWrapper(&(counter(reducedCell)), particle[weighting_] / particles::TYPICAL_NUM_PARTICLE_PER_MAKROPARTICLE);
            }
        }
        __syncthreads();

        if (localId == 0)
        {
            frame = &(pb.getNextFrame(*frame, isValid));
        }
        __syncthreads();
    }


    if (isImageThread)
    {
        image(imageCell) = (Type_) counter(localCell);
    }
}

/**
 * Visualizes simulation data by writing png files.
 * Visulization is performed in an additional thread.
 */
template<class ParticlesType, class Output, typename Type_ = float_X>
class ParticleDensity : public ILightweightPlugin
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;

public:

    typedef Output CreatorType;

    ParticleDensity(std::string name, Output output, uint32_t notifyFrequency, DataSpace<DIM2> transpose, float_X slicePoint) :
    output(output),
    analyzerName(name),
    cellDescription(NULL),
    particleTag(ParticlesType::FrameType::getName()),
    notifyFrequency(notifyFrequency),
    transpose(transpose),
    slicePoint(slicePoint),
    header(NULL),
    isMaster(false)
    {
        sliceDim = 0;
        if (transpose.x() == 0 || transpose.y() == 0)
            sliceDim = 1;
        if ((transpose.x() == 1 || transpose.y() == 1) && sliceDim == 1)
            sliceDim = 2;

        Environment<>::get().PluginConnector().registerPlugin(this);
        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
    }

    virtual ~ParticleDensity()
    {
        if (notifyFrequency > 0)
        {
            __delete(img);
            MessageHeader::destroy(header);
        }
    }

    void notify(uint32_t currentStep)
    {
        const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        Window window(MovingWindow::getInstance().getWindow(currentStep));

        /*sliceOffset is only used in 3D*/
        sliceOffset = (int) ((float) (window.globalDimensions.size[sliceDim]) * slicePoint) +
            window.globalDimensions.offset[sliceDim];

        if (!doDrawing())
        {
            return;
        }
        createImage(currentStep, window);
    }

    void pluginRegisterHelp(po::options_description&)
    {
        // nothing to do here
    }

    std::string pluginGetName() const
    {
        return "ParticleDensity";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    void createImage(uint32_t currentStep, Window window)
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        ParticlesType* particles = &(dc.getData<ParticlesType > (particleTag, true));

        typedef MappingDesc::SuperCellSize SuperCellSize;

        DataSpace<simDim> blockSize(MappingDesc::SuperCellSize::toRT());
        DataSpace<DIM2> blockSize2D(blockSize[transpose.x()], blockSize[transpose.y()]);

        uint32_t globalOffset = 0;
#if(SIMDIM==DIM3)
        globalOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset[sliceDim];
#endif


        //create density image of particles
        __picKernelArea((kernelParticleDensity), *cellDescription, CORE + BORDER)
            (SuperCellSize::toRT().toDim3(), blockSize2D.productOfComponents() * sizeof (float_X))
            (particles->getDeviceParticlesBox(),
             img->getDeviceBuffer().getDataBox(),
             transpose,
             sliceOffset,
             globalOffset, sliceDim
             );


        img->deviceToHost();

        header->update(*cellDescription, window, transpose, currentStep);

        __getTransactionEvent().waitForFinished(); //wait for copy picture

        PMACC_AUTO(hostBox, img->getHostBuffer().getDataBox());

        PMACC_AUTO(resultBox, gather(hostBox, *header));

        // units
        const float_64 cellVolume = CELL_VOLUME;
        float_64 unitVolume = 1.;
        for (uint32_t i = 0; i < simDim; i++)
            unitVolume *= UNIT_LENGTH;
        // that's a hack, but works for all species
        //const float_64 charge = precisionCast<float_64>(
        //    ParticlesType::FrameType().getCharge(particles::TYPICAL_NUM_PARTICLE_PER_MAKROPARTICLE)) /
        //    precisionCast<float_64>(particles::TYPICAL_NUM_PARTICLE_PER_MAKROPARTICLE) * UNIT_CHARGE;

        // Note: multiply particles::TYPICAL_NUM_PARTICLE_PER_MAKROPARTICLE again
        //       because of normalization during atomicAdd above
        //       to avoid float overflow for weightings
        const float_64 unit = precisionCast<float_64>(particles::TYPICAL_NUM_PARTICLE_PER_MAKROPARTICLE)
            / (cellVolume * unitVolume);
        if (isMaster)
            output(resultBox.shift(header->window.offset), unit, header->window.size, *header);
    }

    void init()
    {
        if (notifyFrequency > 0)
        {
            const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

            Window window(MovingWindow::getInstance().getWindow(0));
            sliceOffset = (int) ((float) (window.globalDimensions.size[sliceDim]) * slicePoint) +
                window.globalDimensions.offset[sliceDim];
            const DataSpace<simDim> gpus = Environment<simDim>::get().GridController().getGpuNodes();

            float_32 cellSizeArr[3] = {0, 0, 0};
            for (uint32_t i = 0; i < simDim; ++i)
                cellSizeArr[i] = cellSize[i];
            this->header = MessageHeader::create();
            header->update(*cellDescription, window, transpose, 0, cellSizeArr, gpus);

            img = new GridBuffer<Type_, DIM2 > (header->node.maxSize);

            isMaster = gather.init(doDrawing());
        }
    }

private:

    bool doDrawing()
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        const DataSpace<simDim> globalRootCellPos(subGrid.getLocalDomain().offset);
        const DataSpace<simDim> localSize(subGrid.getLocalDomain().size);
#if(SIMDIM==DIM3)
        const bool tmp = globalRootCellPos[sliceDim] + localSize[sliceDim] > sliceOffset &&
            globalRootCellPos[sliceDim] <= sliceOffset;
        return tmp;
#else
        return true;
#endif
    }

    MappingDesc *cellDescription;
    SimulationDataId particleTag;

    GridBuffer<Type_, DIM2> *img;

    int sliceOffset;
    uint32_t notifyFrequency;
    float_X slicePoint;
    std::string analyzerName;
    DataSpace<DIM2> transpose;
    uint32_t sliceDim;
    MessageHeader *header;
    Output output;
    GatherSlice gather;
    bool isMaster;
};

} //namespace picongpu
