/**
 * Copyright 2016 Heiko Burau
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
#include "math/Vector.hpp"
#include "algorithms/math.hpp"

namespace picongpu
{
using namespace PMacc;

template<typename CalorimeterCur>
struct CalorimeterFunctor
{
    CalorimeterCur calorimeterCur;

    const float_X maxYaw;
    const float_X maxPitch;
    const uint32_t numBinsYaw;
    const uint32_t numBinsPitch;
    const int32_t numBinsEnergy;
    /* depending on `logScale` the energy range is initialized
     * with the logarithmic or the linear value. */
    const float_X minEnergy;
    const float_X maxEnergy;
    const bool logScale;

    const float3_X calorimeterFrameVecX;
    const float3_X calorimeterFrameVecY;
    const float3_X calorimeterFrameVecZ;

    CalorimeterFunctor(const float_X maxYaw,
                       const float_X maxPitch,
                       const uint32_t numBinsYaw,
                       const uint32_t numBinsPitch,
                       const uint32_t numBinsEnergy,
                       const float_X minEnergy,
                       const float_X maxEnergy,
                       const bool logScale,
                       const float3_X calorimeterFrameVecX,
                       const float3_X calorimeterFrameVecY,
                       const float3_X calorimeterFrameVecZ) :
        calorimeterCur(NULL, PMacc::math::Size_t<DIM2>()),
        maxYaw(maxYaw),
        maxPitch(maxPitch),
        numBinsYaw(numBinsYaw),
        numBinsPitch(numBinsPitch),
        numBinsEnergy(numBinsEnergy),
        minEnergy(minEnergy),
        maxEnergy(maxEnergy),
        logScale(logScale),
        calorimeterFrameVecX(calorimeterFrameVecX),
        calorimeterFrameVecY(calorimeterFrameVecY),
        calorimeterFrameVecZ(calorimeterFrameVecZ)
    {}

    HINLINE void setCalorimeterCursor(const CalorimeterCur& calorimeterCur)
    {
        this->calorimeterCur = calorimeterCur;
    }

    template<typename ParticlesFrame>
    DINLINE void operator()(ParticlesFrame& particlesFrame, const uint32_t linearThreadIdx)
    {
        const float3_X mom = particlesFrame[linearThreadIdx][momentum_];
        const float_X mom2 = math::dot(mom, mom);
        float3_X dirVec = mom * math::rsqrt(mom2);

        /* rotate dirVec into the calorimeter frame. This coordinate transformation
         * is performed by a matrix vector multiplication. */
        using namespace PMacc::algorithms::math;
        dirVec = float3_X(dot(this->calorimeterFrameVecX, dirVec),
                          dot(this->calorimeterFrameVecY, dirVec),
                          dot(this->calorimeterFrameVecZ, dirVec));

        /* convert dirVec to yaw and pitch */
        const float_X yaw = atan2(dirVec.x(), dirVec.y());
        const float_X pitch = asin(dirVec.z());

        if(abs(yaw) < this->maxYaw && abs(pitch) < this->maxPitch)
        {
            const float2_X calorimeterPos = particleCalorimeter::mapYawPitchToNormedRange(
                yaw, pitch, this->maxYaw, this->maxPitch);

            // yaw
            int32_t yawBin = calorimeterPos.x() * static_cast<float_X>(numBinsYaw);
            // catch out-of-range values
            yawBin = yawBin >= numBinsYaw ? numBinsYaw - 1 : yawBin;
            yawBin = yawBin < 0 ? 0 : yawBin;

            // pitch
            int32_t pitchBin = calorimeterPos.y() * static_cast<float_X>(numBinsPitch);
            // catch out-of-range values
            pitchBin = pitchBin >= numBinsPitch ? numBinsPitch - 1 : pitchBin;
            pitchBin = pitchBin < 0 ? 0 : pitchBin;

            // energy
            const float_X weighting = particlesFrame[linearThreadIdx][weighting_];
            const float_X normedWeighting = weighting /
                                            static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);
            const float_X energy = SPEED_OF_LIGHT * math::sqrt(mom2) / weighting;

            int32_t energyBin = 0;
            if(this->numBinsEnergy > 1)
            {
                const int32_t numBinsOutOfRange = 2;
                energyBin = math::float2int_rd(((logScale ? log10(energy) : energy) - minEnergy) /
                    (maxEnergy - minEnergy) * static_cast<float_X>(this->numBinsEnergy - numBinsOutOfRange)) + 1;

                // all entries larger than maxEnergy go into last bin
                energyBin = energyBin < this->numBinsEnergy ? energyBin : this->numBinsEnergy - 1;

                // all entries smaller than minEnergy go into bin zero
                energyBin = energyBin > 0 ? energyBin : 0;
            }

            atomicAddWrapper(&(*this->calorimeterCur(yawBin, pitchBin, energyBin)),
                             energy * normedWeighting);
        }
    }
};


template<typename T_ParticlesBox, typename T_CalorimeterFunctor>
struct ParticleCalorimeterKernel
{
    typedef T_ParticlesBox ParticlesBox;
    typedef T_CalorimeterFunctor CalorimeterFunctor;

    ParticlesBox particlesBox;
    CalorimeterFunctor calorimeterFunctor;

    ParticleCalorimeterKernel(ParticlesBox particlesBox,
                           CalorimeterFunctor calorimeterFunctor) :
        particlesBox(particlesBox),
        calorimeterFunctor(calorimeterFunctor)
    {}

    DINLINE void operator()(const PMacc::math::Int<simDim>& cellIndex)
    {
        /* definitions for domain variables, like indices of blocks and threads */
        typedef typename MappingDesc::SuperCellSize SuperCellSize;

        /* multi-dimensional offset vector from local domain origin on GPU in units of super cells */
        const PMacc::math::Int<simDim> block = cellIndex / SuperCellSize::toRT();

        /* multi-dim offset from the origin of the local domain on GPU
         * to the origin of the block of the in unit of cells
         */
        const PMacc::math::Int<simDim> blockCell = block * SuperCellSize::toRT();

        /* multi-dim vector from origin of the block to a cell in units of cells */
        const PMacc::math::Int<simDim> threadIndex = cellIndex % SuperCellSize::toRT();

        /* conversion from a multi-dim cell coordinate to a linear coordinate of the cell in its super cell */
        const int linearThreadIdx = PMacc::math::linearize(
            PMacc::math::CT::shrinkTo<SuperCellSize, simDim-1>::type::toRT(),
            threadIndex);

        typedef typename ParticlesBox::FramePtr ParticlesFramePtr;
        __shared__ typename PMacc::traits::GetEmptyDefaultConstructibleType<ParticlesFramePtr>::type particlesFrame;

        /* find last frame in super cell
         */
        if (linearThreadIdx == 0)
        {
            particlesFrame = this->particlesBox.getLastFrame(block);
        }

        __syncthreads();

        while(particlesFrame.isValid())
        {
            /* casting uint8_t multiMask to boolean */
            const bool isParticle = particlesFrame[linearThreadIdx][multiMask_];

            if(isParticle)
            {
                calorimeterFunctor(particlesFrame, linearThreadIdx);
            }

            __syncthreads();

            if (linearThreadIdx == 0)
            {
                particlesFrame = this->particlesBox.getPreviousFrame(particlesFrame);
            }
            __syncthreads();
        }
    }
};

} // namespace picongpu
