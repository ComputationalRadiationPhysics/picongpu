/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera, Richard Pausch
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
 


#ifndef IMAGE2D_HPP
#define	IMAGE2D_HPP

#include "simulation_defines.hpp"
#include "types.h"


#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"

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
#include <cfloat>


#include "mappings/simulation/GridController.hpp"



#include <string>

#include "memory/boxes/PitchedBox.hpp"

#include "plugins/output/header/MessageHeader.hpp"
#include "plugins/output/GatherSlice.hpp"

#include "algorithms/GlobalReduce.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "nvidia/functors/Max.hpp"

namespace picongpu
{
using namespace PMacc;


// normalize EM fields to typical laser or plasma quantities
//-1: Auto:    enable adaptive scaling for each output
// 1: Laser:   typical fields calculated out of the laser amplitude
// 2: Drift:   typical fields caused by a drifting plasma
// 3: PlWave:  typical fields calculated out of the plasma freq.,
//             assuming the wave moves approx. with c
// 4: Thermal: typical fields calculated out of the electron temperature
// 5: BlowOut: typical fields, assuming that a LWFA in the blowout
//             regime causes a bubble with radius of approx. the laser's
//             beam waist (use for bubble fields)
///  \return float3_X( tyBField, tyEField, tyCurrent )

template< int T >
struct typicalFields
{

    HDINLINE static float3_X get()
    {
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
    }
};

template< >
struct typicalFields < -1 >
{

    HDINLINE static float3_X get()
    {
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
    }
};

template< >
struct typicalFields < 1 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 1 || EM_FIELD_SCALE_CHANNEL2 == 1 || EM_FIELD_SCALE_CHANNEL3 == 1)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X tyCurrent = particleInit::NUM_PARTICLES_PER_CELL * NUM_EL_PER_PARTICLE
            * abs(Q_EL) / DELTA_T;
        const float_X tyEField = laserProfile::AMPLITUDE + FLT_MIN;
        const float_X tyBField = tyEField * MUE0_EPS0;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

template< >
struct typicalFields < 2 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 2 || EM_FIELD_SCALE_CHANNEL2 == 2 || EM_FIELD_SCALE_CHANNEL3 == 2)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        float_X tyCurrent = particleInit::NUM_PARTICLES_PER_CELL * NUM_EL_PER_PARTICLE
            * abs(Q_EL) / DELTA_T;
        // re-normalize currents for non-relativistic drifts
        const double PARTICLE_INIT_DRIFT_BETA =
            sqrt(1.0 - 1.0 / (PARTICLE_INIT_DRIFT_GAMMA *
                              PARTICLE_INIT_DRIFT_GAMMA));
        if (PARTICLE_INIT_DRIFT_GAMMA > 1.0)
            tyCurrent *= float_X(PARTICLE_INIT_DRIFT_BETA);

        const float_X tyBField = MUE0 * tyCurrent;
        const float_X tyEField = tyBField * SPEED_OF_LIGHT;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

template< >
struct typicalFields < 3 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 3 || EM_FIELD_SCALE_CHANNEL2 == 3 || EM_FIELD_SCALE_CHANNEL3 == 3)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X lambda_pl = 2.0f * M_PI * SPEED_OF_LIGHT *
            sqrt(M_EL * EPS0 / GAS_DENSITY / Q_EL / Q_EL);
        const float_X tyEField = lambda_pl * GAS_DENSITY / 3.0f / EPS0;
        const float_X tyBField = tyEField * MUE0_EPS0;
        const float_X tyCurrent = tyBField / MUE0;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

template< >
struct typicalFields < 4 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 4 || EM_FIELD_SCALE_CHANNEL2 == 4 || EM_FIELD_SCALE_CHANNEL3 == 4)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X lambda_deb = sqrt(EPS0 * ELECTRON_TEMPERATURE / GAS_DENSITY
                                        / Q_EL / Q_EL);
        const float_X tyEField = lambda_deb * GAS_DENSITY / 3.0f / EPS0;
        const float_X tyBField = tyEField * MUE0_EPS0;
        const float_X tyCurrent = tyBField / MUE0;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

template< >
struct typicalFields < 5 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 5 || EM_FIELD_SCALE_CHANNEL2 == 5 || EM_FIELD_SCALE_CHANNEL3 == 5)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X tyEField = laserProfile::W0 * GAS_DENSITY / 3.0f / EPS0;
        const float_X tyBField = tyEField * MUE0_EPS0;
        const float_X tyCurrent = particleInit::NUM_PARTICLES_PER_CELL * NUM_EL_PER_PARTICLE
            * abs(Q_EL) / DELTA_T;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

template<class EBox, class BBox, class JBox, class Mapping>
__global__ void kernelPaintFields(
                                  EBox fieldE,
                                  BBox fieldB,
                                  JBox fieldJ,
                                  DataBox<PitchedBox<float3_X, DIM2> > image,
                                  DataSpace<DIM2> transpose,
                                  const int slice,
                                  const uint32_t globalOffset,
                                  const uint32_t sliceDim,
                                  Mapping mapper)
{
    typedef typename MappingDesc::SuperCellSize Block;
    const DataSpace<simDim> threadId(threadIdx);
    const DataSpace<simDim> block = mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx));
    const DataSpace<simDim> cell(block * Block::getDataSpace() + threadId);
    const DataSpace<simDim> blockOffset((block - mapper.getGuardingSuperCells()) * Block::getDataSpace());


    const DataSpace<simDim> realCell(cell - MappingDesc::SuperCellSize::getDataSpace() * mapper.getGuardingSuperCells()); //delete guard from cell idx
    const DataSpace<DIM2> imageCell(
                                    realCell[transpose.x()],
                                    realCell[transpose.y()]);
    const DataSpace<simDim> realCell2(blockOffset + threadId); //delete guard from cell idx

    uint32_t globalCell = realCell2[sliceDim] + globalOffset;

    if (globalCell != slice)
        return;

    // set fields of this cell to vars
    typename BBox::ValueType field_b = fieldB(cell);
    typename EBox::ValueType field_e = fieldE(cell);
    typename JBox::ValueType field_j = fieldJ(cell);
    field_j = float3_X(
                       field_j.x() * CELL_HEIGHT * CELL_DEPTH,
                       field_j.y() * CELL_WIDTH * CELL_DEPTH,
                       field_j.z() * CELL_WIDTH * CELL_HEIGHT
                       );
    
    // reset picture to black
    //   color range for each RGB channel: [0.0, 1.0]
    float3_X pic = float3_X(0., 0., 0.);

    // typical values of the fields to normalize them to [0,1]
    //
    pic.x() = visPreview::preChannel1(field_b / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().x(),
                                      field_e / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().y(),
                                      field_j / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().z());
    pic.y() = visPreview::preChannel2(field_b / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().x(),
                                      field_e / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().y(),
                                      field_j / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().z());
    pic.z() = visPreview::preChannel3(field_b / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().x(),
                                      field_e / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().y(),
                                      field_j / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().z());
    //visPreview::preChannel1Col::addRGB(pic,
    //                                   visPreview::preChannel1(field_b * typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().x(),
    //                                                           field_e * typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().y(),
    //                                                           field_j * typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().z()),
    //                                   visPreview::preChannel1_opacity);
    //visPreview::preChannel2Col::addRGB(pic,
    //                                   visPreview::preChannel2(field_b * typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().x(),
    //                                                           field_e * typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().y(),
    //                                                           field_j * typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().z()),
    //                                   visPreview::preChannel2_opacity);
    //visPreview::preChannel3Col::addRGB(pic,
    //                                   visPreview::preChannel3(field_b * typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().x(),
    //                                                           field_e * typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().y(),
    //                                                           field_j * typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().z()),
    //                                   visPreview::preChannel3_opacity);


    // draw to (perhaps smaller) image cell
    image(imageCell) = pic;
}

template<class ParBox, class Mapping>
__global__ void
kernelPaintParticles3D(ParBox pb,
                       DataBox<PitchedBox<float3_X, DIM2> > image,
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
        atomicExch((int*) &isValid, 1); /*WAW Error in cuda-memcheck racecheck*/
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
        /** Note: normally, we would multiply by NUM_EL_PER_PARTICLE again.
         *  BUT: since we are interested in a simple value between 0 and 1,
         *       we stay with this number (normalized to the order of macro
         *       particles) and devide by the number of typical macro particles
         *       per cell
         */
        float_X value = counter(localCell)
            / float_X(particleInit::NUM_PARTICLES_PER_CELL); // * NUM_EL_PER_PARTICLE;
        if (value > 1.0) value = 1.0;

        //image(imageCell).x() = value;
        visPreview::preParticleDensCol::addRGB(image(imageCell),
                                               value,
                                               visPreview::preParticleDens_opacity);

        // cut to [0, 1]
        if (image(imageCell).x() < float_X(0.0)) image(imageCell).x() = float_X(0.0);
        if (image(imageCell).x() > float_X(1.0)) image(imageCell).x() = float_X(1.0);
        if (image(imageCell).y() < float_X(0.0)) image(imageCell).y() = float_X(0.0);
        if (image(imageCell).y() > float_X(1.0)) image(imageCell).y() = float_X(1.0);
        if (image(imageCell).z() < float_X(0.0)) image(imageCell).z() = float_X(0.0);
        if (image(imageCell).z() > float_X(1.0)) image(imageCell).z() = float_X(1.0);
    }
}


namespace vis_kernels
{

template<class Mem, typename Type>
__global__ void divideAnyCell(Mem mem, uint32_t n, Type divisor)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const float3_X FLT3_MIN = float3_X(FLT_MIN, FLT_MIN, FLT_MIN);
    mem[tid] /= (divisor + FLT3_MIN);
}

template<class Mem>
__global__ void channelsToRGB(Mem mem, uint32_t n)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float3_X rgb = float3_X(float_X(0.0), float_X(0.0), float_X(0.0));

    visPreview::preChannel1Col::addRGB(rgb,
                                       mem[tid].x(),
                                       visPreview::preChannel1_opacity);
    visPreview::preChannel2Col::addRGB(rgb,
                                       mem[tid].y(),
                                       visPreview::preChannel2_opacity);
    visPreview::preChannel3Col::addRGB(rgb,
                                       mem[tid].z(),
                                       visPreview::preChannel3_opacity);
    mem[tid] = rgb;
}

}

/**
 * Visualizes simulation data by writing png files.
 * Visulization is performed in an additional thread.
 */
template<class ParticlesType, class Output>
class Visualisation : public ISimulationIO
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;


public:
    typedef Output CreatorType;

    Visualisation(std::string name, Output output, uint32_t notifyFrequency, DataSpace<DIM2> transpose, float_X slicePoint) :
    output(output),
    analyzerName(name),
    cellDescription(NULL),
    particleTag(ParticlesType::FrameType::CommunicationTag),
    notifyFrequency(notifyFrequency),
    transpose(transpose),
    slicePoint(slicePoint),
    isMaster(false),
    reduce(1024)
    {
        sliceDim = 0;
        if (transpose.x() == 0 || transpose.y() == 0)
            sliceDim = 1;
        if ((transpose.x() == 1 || transpose.y() == 1) && sliceDim == 1)
            sliceDim = 2;
    }

    virtual ~Visualisation()
    {
        if (notifyFrequency > 0)
        {
            __delete(img);
        }
    }

    void notify(uint32_t currentStep)
    {
        assert(cellDescription != NULL);
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
        assert(cellDescription != NULL);
        this->cellDescription = cellDescription;
    }

    void createImage(uint32_t currentStep, VirtualWindow window)
    {
        DataConnector &dc = DataConnector::getInstance();
        // Data does not need to be synchronized as visualization is
        // done at the device.
        FieldB *fieldB = &(dc.getData<FieldB > (FIELD_B, true));
        FieldE* fieldE = &(dc.getData<FieldE > (FIELD_E, true));
        FieldJ* fieldJ = &(dc.getData<FieldJ > (FIELD_J, true));
        ParticlesType* particles = &(dc.getData<ParticlesType > (particleTag, true));
        
        PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());

        typedef MappingDesc::SuperCellSize SuperCellSize;
        assert(cellDescription != NULL);
        //create image fields
        __picKernelArea((kernelPaintFields), *cellDescription, CORE + BORDER)
            (SuperCellSize::getDataSpace())
            (fieldE->getDeviceDataBox(),
             fieldB->getDeviceDataBox(),
             fieldJ->getDeviceDataBox(),
             img->getDeviceBuffer().getDataBox(),
             transpose,
             sliceOffset,
             simBox.getGlobalOffset()[sliceDim], sliceDim
             );

        // find maximum for img.x()/y and z and return it as float3_X
        int elements = img->getGridLayout().getDataSpace().getElementCount();

        //Add one dimension access to 2d DataBox
        typedef DataBoxDim1Access<typename GridBuffer<float3_X, DIM2 >::DataBoxType> D1Box;
        D1Box d1access(img->getDeviceBuffer().getDataBox(), img->getGridLayout().getDataSpace());

#if (EM_FIELD_SCALE_CHANNEL1 == -1 || EM_FIELD_SCALE_CHANNEL2 == -1 || EM_FIELD_SCALE_CHANNEL3 == -1)
        //reduce with functor max
        float3_X max = reduce(nvidia::functors::Max(),
                              d1access,
                              elements);
        //reduce with functor min
        //float3_X min = reduce(nvidia::functors::Min(),
        //                    d1access,
        //                    elements);
#if (EM_FIELD_SCALE_CHANNEL1 != -1 )
        max.x() = float_X(1.0);
#endif
#if (EM_FIELD_SCALE_CHANNEL2 != -1 )
        max.y() = float_X(1.0);
#endif
#if (EM_FIELD_SCALE_CHANNEL3 != -1 )
        max.z() = float_X(1.0);
#endif

        //We don't know the superCellSize at compile time 
        // (because of the runtime dimension selection in any analyser), 
        // thus we must use a one dimension kernel and no mapper
        __cudaKernel(vis_kernels::divideAnyCell)(ceil((double) elements / 256), 256)(d1access, elements, max);
#endif

        // convert channels to RGB
        __cudaKernel(vis_kernels::channelsToRGB)(ceil((double) elements / 256), 256)(d1access, elements);

        // add density color channel
        DataSpace<simDim> blockSize(MappingDesc::SuperCellSize::getDataSpace());
        DataSpace<DIM2> blockSize2D(blockSize[transpose.x()], blockSize[transpose.y()]);

        //create image particles
        __picKernelArea((kernelPaintParticles3D), *cellDescription, CORE + BORDER)
            (SuperCellSize::getDataSpace(), blockSize2D.getElementCount() * sizeof (int))
            (particles->getDeviceParticlesBox(),
             img->getDeviceBuffer().getDataBox(),
             transpose,
             sliceOffset,
             simBox.getGlobalOffset()[sliceDim], sliceDim
             );

        // send the RGB image back to host
        img->deviceToHost();


        header.update(*cellDescription, window, transpose, currentStep);


        __getTransactionEvent().waitForFinished(); //wait for copy picture

        DataSpace<DIM2> size = img->getGridLayout().getDataSpace();

        PMACC_AUTO(hostBox, img->getHostBuffer().getDataBox());

        if (picongpu::white_box_per_GPU)
        {
            hostBox[0 ][0 ] = float3_X(1.0, 1.0, 1.0);
            hostBox[size.y() - 1 ][0 ] = float3_X(1.0, 1.0, 1.0);
            hostBox[0 ][size.x() - 1] = float3_X(1.0, 1.0, 1.0);
            hostBox[size.y() - 1 ][size.x() - 1] = float3_X(1.0, 1.0, 1.0);
        }
        PMACC_AUTO(resultBox, gather(hostBox, header));
        if (isMaster)
        {
            output(resultBox.shift(header.window.offset), header.window.size, header);
        }

    }

    void init()
    {
        if (notifyFrequency > 0)
        {
            assert(cellDescription != NULL);
            const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

            VirtualWindow window(MovingWindow::getInstance().getVirtualWindow(0));
            sliceOffset = (int) ((float) (window.globalWindowSize[sliceDim]) * slicePoint) + window.globalSimulationOffset[sliceDim];


            const DataSpace<simDim> gpus = GridController<simDim>::getInstance().getGpuNodes();

            float_32 cellSize[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};

            header.update(*cellDescription, window, transpose, 0, cellSize, gpus);

            img = new GridBuffer<float3_X, DIM2 > (header.node.maxSize);


            DataConnector::getInstance().registerObserver(this, notifyFrequency);

            bool isDrawing = doDrawing();
            isMaster = gather.init(isDrawing);
            reduce.participate(isDrawing);

        }
    }

private:

    bool doDrawing()
    {
        assert(cellDescription != NULL);
        const DataSpace<simDim> globalRootCellPos(SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset());
        const bool tmp = globalRootCellPos[sliceDim] + SubGrid<simDim>::getInstance().getSimulationBox().getLocalSize()[sliceDim] > sliceOffset &&
            globalRootCellPos[sliceDim] <= sliceOffset;
        /* std::cout << "do drawing ." << tmp << "." <<
                 globalRootCellPos[sliceDim] + cellDescription->getGridLayout().getDataSpaceWithoutGuarding()[sliceDim] << ">" << sliceOffset <<
                 "&&" << globalRootCellPos[sliceDim] << "<=" << sliceOffset << std::endl;*/
        return tmp;
    }


    MappingDesc *cellDescription;
    uint32_t particleTag;

    GridBuffer<float3_X, DIM2 > *img;

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
    algorithms::GlobalReduce reduce;
};



}


#endif	/* IMAGE2D_HPP */
