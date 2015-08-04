/**
 * Copyright 2013-2015 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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

#include <string>
#include <iostream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "algorithms/Gamma.hpp"
#include "plugins/ILightweightPlugin.hpp"

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

    SglParticle() : position(FloatPos::create(0.0)), momentum(float3_X::create(0.0)), mass(0.0),
        weighting(0.0), charge(0.0), gamma(0.0)
    {
    }

    DataSpace<simDim> globalCellOffset;

    //! todo

    floatD_64 getGlobalCell() const
    {
        floatD_64 doubleGlobalCellOffset;
        for(uint32_t i=0;i<simDim;++i)
            doubleGlobalCellOffset[i]=float_64(globalCellOffset[i]);

        return floatD_64( doubleGlobalCellOffset + precisionCast<float_64>(position));
    }

    template<typename T>
        friend std::ostream& operator<<(std::ostream& out, const SglParticle<T>& v)
    {
        floatD_64 pos;
        for(uint32_t i=0;i<simDim;++i)
            pos[i]=( v.getGlobalCell()[i] * cellSize[i]*UNIT_LENGTH);

        const float3_64 mom( precisionCast<float_64>(v.momentum.x()) * UNIT_MASS * UNIT_SPEED,
                             precisionCast<float_64>(v.momentum.y()) * UNIT_MASS * UNIT_SPEED,
                             precisionCast<float_64>(v.momentum.z()) * UNIT_MASS * UNIT_SPEED );

        const float_64 mass = precisionCast<float_64>(v.mass) * UNIT_MASS;
        const float_64 charge = precisionCast<float_64>(v.charge) * UNIT_CHARGE;

        typedef std::numeric_limits< float_64 > dbl;
        out.precision(dbl::digits10);

        out << std::scientific << pos << " " << mom << " " << mass << " "
            << precisionCast<float_64>(v.weighting)
            << " " << charge << " " << precisionCast<float_64>(v.gamma);
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

    /* BUGFIX to issue #538
     * volatile prohibits that the compiler creates wrong code*/
    volatile bool isParticle = (*frame)[linearThreadIdx][multiMask_];

    while (isValid)
    {
        if (isParticle)
        {
            PMACC_AUTO(particle,(*frame)[linearThreadIdx]);
            gParticle->position = particle[position_];
            gParticle->momentum = particle[momentum_];
            gParticle->weighting = particle[weighting_];
            gParticle->mass = attribute::getMass(gParticle->weighting,particle);
            gParticle->charge = attribute::getCharge(gParticle->weighting,particle);
            gParticle->gamma = Gamma<>()(gParticle->momentum, gParticle->mass);

            // storage number in the actual frame
            const lcellId_t frameCellNr = particle[localCellIdx_];

            // offset in the actual superCell = cell offset in the supercell
            const DataSpace<simDim> frameCellOffset(DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize > (frameCellNr));


            gParticle->globalCellOffset = (superCellIdx - mapper.getGuardingSuperCells())
                * MappingDesc::SuperCellSize::toRT()
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
class PositionsParticles : public ILightweightPlugin
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;
    typedef floatD_X FloatPos;

    ParticlesType *particles;

    GridBuffer<SglParticle<FloatPos>, DIM1> *gParticle;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;

    std::string analyzerName;
    std::string analyzerPrefix;

public:

    PositionsParticles() :
    analyzerName("PositionsParticles: write position of one particle of a species to std::cout"),
    analyzerPrefix(ParticlesType::FrameType::getName() + std::string("_position")),
    particles(NULL),
    gParticle(NULL),
    cellDescription(NULL),
    notifyFrequency(0)
    {

        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~PositionsParticles()
    {
    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));


        const int rank = Environment<simDim>::get().GridController().getGlobalRank();
        const SglParticle<FloatPos> positionParticle = getPositionsParticles < CORE + BORDER > (currentStep);

        /*FORMAT OUTPUT*/
        if (positionParticle.mass != float_X(0.0))
            std::cout << "[ANALYSIS] [" << rank << "] [COUNTER] [" << analyzerPrefix << "] [" << currentStep << "] "
            << std::setprecision(16) << float_64(currentStep) * SI::DELTA_T_SI << " "
            << positionParticle << "\n"; // no flush
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]");
    }

    std::string pluginGetName() const
    {
        return analyzerName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    void pluginLoad()
    {
        if (notifyFrequency > 0)
        {
            //create one float3_X on gpu und host
            gParticle = new GridBuffer<SglParticle<FloatPos>, DIM1 > (DataSpace<DIM1 > (1));

            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
        }
    }

    void pluginUnload()
    {
        __delete(gParticle);
    }

    template< uint32_t AREA>
    SglParticle<FloatPos> getPositionsParticles(uint32_t currentStep)
    {

        typedef typename MappingDesc::SuperCellSize SuperCellSize;
        SglParticle<FloatPos> positionParticleTmp;

        gParticle->getDeviceBuffer().setValue(positionParticleTmp);
        dim3 block(SuperCellSize::toRT().toDim3());

        __picKernelArea(kernelPositionsParticles, *cellDescription, AREA)
            (block)
            (particles->getDeviceParticlesBox(),
             gParticle->getDeviceBuffer().getBasePointer());
        gParticle->deviceToHost();

        DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);

        DataSpace<simDim> gpuPhyCellOffset(Environment<simDim>::get().SubGrid().getLocalDomain().offset);
        gpuPhyCellOffset.y() += (localSize.y() * numSlides);

        gParticle->getHostBuffer().getDataBox()[0].globalCellOffset += gpuPhyCellOffset;


        return gParticle->getHostBuffer().getDataBox()[0];
    }

};

}
