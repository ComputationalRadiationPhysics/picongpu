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
 


#ifndef POSITIONSPARTICLES_HPP
#define	POSITIONSPARTICLES_HPP

#include <string>
#include <iostream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "basicOperations.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "algorithms/Gamma.hpp"
#include "plugins/IPluginModule.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

template<class FloatPos>
struct SglParticle
{
    FloatPos position;
    float3_X momentum;
    float_X mass;
    float_X weighting;
    float_X charge;
    float_X gamma;

    SglParticle() : position(0.0,0.0,0.0), momentum(0.0,0.0, 0.0), mass(0.0),
        weighting(0.0), charge(0.0), gamma(0.0)
    {
    }

    DataSpace<simDim> globalCellOffset;

    //! todo 

    float3_64 getGlobalCell() const
    {
        return float3_64( typeCast<float_64>(globalCellOffset.x()) + typeCast<float_64>(position.x()),
                          typeCast<float_64>(globalCellOffset.y()) + typeCast<float_64>(position.y()),
                          typeCast<float_64>(globalCellOffset.z()) + typeCast<float_64>(position.z()) );
    }

    template<typename T>
        friend std::ostream& operator<<(std::ostream& out, const SglParticle<T>& v)
    {
        const float3_64 pos( v.getGlobalCell().x() * SI::CELL_WIDTH_SI,
                             v.getGlobalCell().y() * SI::CELL_HEIGHT_SI,
                             v.getGlobalCell().z() * SI::CELL_DEPTH_SI   );
        const float3_64 mom( typeCast<float_64>(v.momentum.x()) * UNIT_MASS * UNIT_SPEED,
                             typeCast<float_64>(v.momentum.y()) * UNIT_MASS * UNIT_SPEED,
                             typeCast<float_64>(v.momentum.z()) * UNIT_MASS * UNIT_SPEED );
        
        const float_64 mass = typeCast<float_64>(v.mass) * UNIT_MASS;
        const float_64 charge = typeCast<float_64>(v.charge) * UNIT_CHARGE;

        typedef std::numeric_limits< float_64 > dbl;
        out.precision(dbl::digits10);

        out << std::scientific << pos << " " << mom << " " << mass << " "
            << typeCast<float_64>(v.weighting)
            << " " << charge << " " << typeCast<float_64>(v.gamma);
        return out;
    }
};

/** write the position of a single particle to a file
 * \warning this analyser MUST NOT be used with more than one (global!)
 * particle and is created for one-particle-test-purposes only
 */
template<class FRAME, class FloatPos, class Mapping>
__global__ void kernelPositionsParticles(ParticlesBox<FRAME, simDim> pb,
                                         SglParticle<FloatPos>* gParticle,
                                         Mapping mapper)
{

    __shared__ FRAME *frame;
    __shared__ bool isValid;
    __syncthreads(); /*wait that all shared memory is initialised*/

    typedef typename Mapping::SuperCellSize SuperCellSize;

    const DataSpace<simDim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);
    const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));

    if (linearThreadIdx == 0)
    {
        frame = &(pb.getLastFrame(superCellIdx, isValid));
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

    bool isParticle = (*frame)[linearThreadIdx][multiMask_];

    while (isValid)
    {
        if (isParticle)
        {
            PMACC_AUTO(particle,(*frame)[linearThreadIdx]);
            gParticle->position = particle[position_];
            gParticle->momentum = particle[momentum_];
            gParticle->weighting = particle[weighting_];
            gParticle->mass = frame->getMass(gParticle->weighting);
            gParticle->charge = frame->getCharge(gParticle->weighting);
            gParticle->gamma = Gamma<>()(gParticle->momentum, gParticle->mass);

            // storage number in the actual frame
            const lcellId_t frameCellNr = particle[localCellIdx_];

            // offset in the actual superCell = cell offset in the supercell
            const DataSpace<simDim> frameCellOffset(DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize > (frameCellNr));


            gParticle->globalCellOffset = (superCellIdx - mapper.getGuardingSuperCells())
                * MappingDesc::SuperCellSize::getDataSpace()
                + frameCellOffset;
        }
        __syncthreads();
        if (linearThreadIdx == 0)
        {
            frame = &(pb.getPreviousFrame(*frame, isValid));
        }
        isParticle = true;
        __syncthreads();
    }

}

template<class ParticlesType>
class PositionsParticles : public ISimulationIO, public IPluginModule
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;
    typedef float3_X FloatPos;

    ParticlesType *particles;

    GridBuffer<SglParticle<FloatPos>, DIM1> *gParticle;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;

    std::string analyzerName;
    std::string analyzerPrefix;

public:

    PositionsParticles(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    particles(NULL),
    gParticle(NULL),
    cellDescription(NULL),
    notifyFrequency(0)
    {

        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~PositionsParticles()
    {
    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = DataConnector::getInstance();

        particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));


        const int rank = GridController<simDim>::getInstance().getGlobalRank();
        const SglParticle<FloatPos> positionParticle = getPositionsParticles < CORE + BORDER > (currentStep);

        /*FORMAT OUTPUT*/
        if (positionParticle.mass != float_X(0.0))
            std::cout << "[ANALYSIS] [" << rank << "] [COUNTER] [" << analyzerName << "] [" << currentStep << "] "
            << std::setprecision(16) << double(currentStep) * SI::DELTA_T_SI << " "
            << positionParticle << "\n"; // no flush
    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]");
    }

    std::string moduleGetName() const
    {
        return analyzerName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            //create one float3_X on gpu und host
            gParticle = new GridBuffer<SglParticle<FloatPos>, DIM1 > (DataSpace<DIM1 > (1));

            DataConnector::getInstance().registerObserver(this, notifyFrequency);
        }
    }

    void moduleUnload()
    {
        __delete(gParticle);
    }

    template< uint32_t AREA>
    SglParticle<FloatPos> getPositionsParticles(uint32_t currentStep)
    {

        typedef typename MappingDesc::SuperCellSize SuperCellSize;
        SglParticle<FloatPos> positionParticleTmp;

        gParticle->getDeviceBuffer().setValue(positionParticleTmp);
        dim3 block(SuperCellSize::getDataSpace());

        __picKernelArea(kernelPositionsParticles, *cellDescription, AREA)
            (block)
            (particles->getDeviceParticlesBox(),
             gParticle->getDeviceBuffer().getBasePointer());
        gParticle->deviceToHost();

        DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        VirtualWindow window(MovingWindow::getInstance().getVirtualWindow(currentStep));

        DataSpace<simDim> gpuPhyCellOffset(SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset());
        gpuPhyCellOffset.y() += (localSize.y() * window.slides);

        gParticle->getHostBuffer().getDataBox()[0].globalCellOffset += gpuPhyCellOffset;


        return gParticle->getHostBuffer().getDataBox()[0];
    }

};

}


#endif	/* POSITIONSPARTICLES_HPP */

