/**
 * Copyright 2014 Rene Widera
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
#include "nvidia/atomic.hpp"

namespace picongpu
{
namespace particles
{
namespace manipulators
{

struct Assign
{

    HINLINE Assign(const uint32_t)
    {
    }

    template<typename T_DestParticle, typename T_SrcParticle>
    HDINLINE void operator()(T_DestParticle& destPar, const T_SrcParticle& srcPar)
    {
        PMacc::particles::operations::assign(destPar, srcPar);
    }

};

template<typename T_Functor, typename T_DestSpeciesType, typename T_Count, typename T_SrcSpeciesType>
struct CreateParticlesFromParticleImpl : private T_Functor
{
    typedef T_DestSpeciesType DestSpeciesType;
    typedef T_SrcSpeciesType SrcSpeciesType;
    typedef typename MakeIdentifier<SrcSpeciesType>::type SpeciesName;
    typedef typename MakeIdentifier<DestSpeciesType>::type DestSpeciesName;


    typedef T_Functor Functor;
    static const uint32_t particlePerParticle = T_Count::value;
    static const int cellsInSuperCell = (int)PMacc::math::CT::volume<SuperCellSize>::type::value;

    HINLINE CreateParticlesFromParticleImpl(uint32_t currentStep) : Functor(currentStep)
    {
        typedef typename DestSpeciesType::FrameType DestFrameType;
        typedef typename DestSpeciesType::ParticlesBoxType DestParticlesBoxType;
        guardCells = SuperCellSize::toRT(); //\todo: ask mapper or any other class
        DataConnector &dc = Environment<>::get().DataConnector();
        DestSpeciesType& destSpecies = dc.getData<DestSpeciesType > (DestFrameType::getName(), true);
        destParBox = destSpecies.getDeviceParticlesBox();
        firstCall = true;
    }

    template<typename T_Particle1, typename T_Particle2>
    DINLINE void operator()(const DataSpace<simDim>& localCellIdx,
                            T_Particle1& particle, T_Particle2&,
                            const bool isParticle, const bool)
    {
        typedef typename DestSpeciesType::FrameType DestFrameType;
        typedef typename DestSpeciesType::ParticlesBoxType DestParticlesBoxType;
        typedef typename SrcSpeciesType::FrameType FrameType;

        typedef typename FrameType::ParticleType SrcParticleType;
        typedef typename DestFrameType::ParticleType DestParticleType;

        __shared__ DestFrameType* destFrame;
        __shared__ int particlesInDestSuperCell;

        __syncthreads();

        uint32_t ltid = DataSpaceOperations<simDim>::template map<SuperCellSize>(DataSpace<simDim>(threadIdx));
        const DataSpace<simDim> superCell((guardCells + localCellIdx) / SuperCellSize::toRT());
        if (ltid == 0)
        {
            if (firstCall)
            {
                bool isValid;

                destFrame = &(destParBox.getLastFrame(superCell, isValid));
                particlesInDestSuperCell = 0;
                if (isValid)
                {
                    particlesInDestSuperCell = destParBox.getSuperCell(superCell).getSizeLastFrame();
                }
                if (!isValid || particlesInDestSuperCell == cellsInSuperCell)
                {
                    destFrame = &(destParBox.getEmptyFrame());
                    destParBox.setAsLastFrame(*destFrame, superCell);
                }
                firstCall = false;
            }
        }
        __syncthreads();


        int numParToCreate = particlePerParticle;
        int oldParCounter;
       // int newParCounter;

        while (true)
        {
            __syncthreads();
            int freeSlot = -1;
            oldParCounter = particlesInDestSuperCell;
            __syncthreads();
            if (isParticle && numParToCreate > 0)
            {
                freeSlot = nvidia::atomicAllInc(&particlesInDestSuperCell);
            }
            --numParToCreate;
            if (freeSlot>-1 && freeSlot < cellsInSuperCell)
            {
                PMACC_AUTO(destParticle, (*destFrame)[freeSlot]);
                Functor::operator()(destParticle, particle);
            }
            __syncthreads();
            if(oldParCounter == particlesInDestSuperCell)
                break;
            __syncthreads();

            if (ltid == 0)
            {
                if (particlesInDestSuperCell >= cellsInSuperCell)
                {
                    particlesInDestSuperCell -= cellsInSuperCell;
                    destFrame = &(destParBox.getEmptyFrame());
                    destParBox.setAsLastFrame(*destFrame, superCell);
                }
            }
            __syncthreads();

            //second flush
            if (freeSlot >= cellsInSuperCell)
            {
                PMACC_AUTO(destParticle, (*destFrame)[freeSlot - cellsInSuperCell]);
                Functor::operator()(destParticle, particle);
            }
            __syncthreads();
        }

    }

private:

    DataSpace<simDim> guardCells;
    typename traits::GetDataBoxType<DestSpeciesType>::type destParBox;
    bool firstCall;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
