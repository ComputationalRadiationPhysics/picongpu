/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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
 


#ifndef PARTICLEDENSITY_HPP
#define	PARTICLEDENSITY_HPP

#include "simulation_defines.hpp"
#include "types.h"

#include "dimensions/TVec.h"
#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "memory/buffers/GridBuffer.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"

#include "dataManagement/DataConnector.hpp"
#include "dataManagement/ISimulationIO.hpp"
#include "dimensions/TVec.h"

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
    const DataSpace<simDim> blockOffset((block - 1) * Block::getDataSpace());


    int localId = threadIdx.z * Block::x * Block::y + threadIdx.y * Block::x + threadIdx.x;


    if (localId == 0)
        isValid = false;
    __syncthreads();

    //\todo: guard size should not be set to (fixed) 1 here
    const DataSpace<simDim> realCell(blockOffset + threadId); //delete guard from cell idx


    uint32_t globalCell = realCell[sliceDim] + globalOffset;

    if (globalCell == slice)
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
        PMACC_AUTO(particle,(*frame)[localId]);
        if (particle[multiMask_] == 1)
        {
            int cellIdx = particle[localCellIdx_];
            // we only draw the first slice of cells in the super cell (z == 0)
            const DataSpace<DIM3> particleCellId(DataSpaceOperations<DIM3>::template map<Block > (cellIdx));
            uint32_t globalParticleCell = particleCellId[sliceDim] + globalOffset + blockOffset[sliceDim];
            if (globalParticleCell == slice)
            {
                const DataSpace<DIM2> reducedCell(particleCellId[transpose.x()], particleCellId[transpose.y()]);
                atomicAddWrapper(&(counter(reducedCell)), particle[weighting_] / NUM_EL_PER_PARTICLE);
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
class ParticleDensity : public ISimulationIO
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;

public:

    typedef Output CreatorType;

    ParticleDensity(std::string name, Output output, uint32_t notifyFrequency, DataSpace<DIM2> transpose, float_X slicePoint) :
    output(output),
    analyzerName(name),
    cellDescription(NULL),
    particleTag(ParticlesType::FrameType::CommunicationTag),
    notifyFrequency(notifyFrequency),
    transpose(transpose),
    slicePoint(slicePoint),
    isMaster(false)
    {
        sliceDim = 0;
        if (transpose.x() == 0 || transpose.y() == 0)
            sliceDim = 1;
        if ((transpose.x() == 1 || transpose.y() == 1) && sliceDim == 1)
            sliceDim = 2;
    }

    virtual ~ParticleDensity()
    {
        if (notifyFrequency > 0)
        {
            __delete(img);
        }
    }

    void notify(uint32_t currentStep)
    {
        const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        VirtualWindow window(MovingWindow::getInstance().getVirtualWindow(currentStep));

        sliceOffset = (int) ((float) (window.globalWindowSize[sliceDim]) * slicePoint) + window.globalSimulationOffset[sliceDim];

        if (!doDrawing())
        {
            return;
        }
        createImage(currentStep, window);
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    void createImage(uint32_t currentStep, VirtualWindow window)
    {
        DataConnector &dc = DataConnector::getInstance();
        ParticlesType* particles = &(dc.getData<ParticlesType > (particleTag, true));

        typedef MappingDesc::SuperCellSize SuperCellSize;

        DataSpace<simDim> blockSize(MappingDesc::SuperCellSize::getDataSpace());
        DataSpace<DIM2> blockSize2D(blockSize[transpose.x()], blockSize[transpose.y()]);

        //create density image of particles
        __picKernelArea((kernelParticleDensity), *cellDescription, CORE + BORDER)
            (SuperCellSize::getDataSpace(), blockSize2D.getElementCount() * sizeof (float_X))
            (particles->getDeviceParticlesBox(),
             img->getDeviceBuffer().getDataBox(),
             transpose,
             sliceOffset,
             (SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset())[sliceDim], sliceDim
             );


        img->deviceToHost();

        header.update(*cellDescription, window, transpose, currentStep);

        __getTransactionEvent().waitForFinished(); //wait for copy picture

        PMACC_AUTO(hostBox, img->getHostBuffer().getDataBox());

        PMACC_AUTO(resultBox, gather(hostBox, header));
        
        // units
        const float_64 cellVolume = CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH;
        const float_64 unitVolume = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
        // that's a hack, but works for all species
        //const float_64 charge = typeCast<float_64>(
        //    ParticlesType::FrameType().getCharge(NUM_EL_PER_PARTICLE)) /
        //    typeCast<float_64>(NUM_EL_PER_PARTICLE) * UNIT_CHARGE;
        
        // Note: multiply NUM_EL_PER_PARTICLE again
        //       because of normalization during atomicAdd above
        //       to avoid float overflow for weightings
        const float_64 unit = typeCast<float_64>(NUM_EL_PER_PARTICLE)
                            / ( cellVolume * unitVolume );
                            // * charge
        if (isMaster)
            output(resultBox.shift(header.window.offset), unit, header.window.size, header);
    }

    void init()
    {
        if (notifyFrequency > 0)
        {
            const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

            DataConnector::getInstance().registerObserver(this, notifyFrequency);

            VirtualWindow window(MovingWindow::getInstance().getVirtualWindow(0));
            sliceOffset = (int) ((float) (window.globalWindowSize[sliceDim]) * slicePoint) + window.globalSimulationOffset[sliceDim];
            const DataSpace<simDim> gpus = GridController<simDim>::getInstance().getGpuNodes();

            float_32 cellSize[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
            header.update(*cellDescription, window, transpose, 0, cellSize, gpus);

            img = new GridBuffer<Type_, DIM2 > (header.node.maxSize);

            isMaster = gather.init(doDrawing());
        }
    }

private:

    bool doDrawing()
    {
        PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());
        const DataSpace<simDim> globalRootCellPos(simBox.getGlobalOffset());
        const DataSpace<simDim> localSize(simBox.getLocalSize());
        const bool tmp = globalRootCellPos[sliceDim] + localSize[sliceDim] > sliceOffset &&
            globalRootCellPos[sliceDim] <= sliceOffset;
        return tmp;
    }


    MappingDesc *cellDescription;
    uint32_t particleTag;

    GridBuffer<Type_, DIM2> *img;

    int sliceOffset;
    uint32_t notifyFrequency;
    float_X slicePoint;

    std::string analyzerName;


    DataSpace<DIM2> transpose;
    uint32_t sliceDim;

    MessageHeader header;

    Output output;
    GatherSlice gather;
    bool isMaster;
};



}


#endif	/* PARTICLEDENSITY_HPP */
