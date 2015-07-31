/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera
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

#include "simulation_defines.hpp"
#include "types.h"

#include "math/Vector.hpp"
#include "particles/particleToGrid/ComputeGridValuePerFrame.def"

#include "algorithms/Gamma.hpp"

namespace picongpu
{
namespace particleToGrid
{

template<class T_ParticleShape, uint32_t calcType>
HDINLINE float1_64
ComputeGridValuePerFrame<T_ParticleShape, calcType>::getUnit() const
{
    const float_64 UNIT_VOLUME = (UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH);

    if (calcType == ComputeGridValueOptions::calcDensity)
        return UNIT_CHARGE / UNIT_VOLUME;
    else
        if (calcType == ComputeGridValueOptions::calcEnergy)
        return UNIT_ENERGY;
    else
        if (calcType == ComputeGridValueOptions::calcEnergyDensity)
        return UNIT_CHARGE / UNIT_VOLUME * UNIT_ENERGY;
    else
        if (calcType == ComputeGridValueOptions::calcCounter)
        return particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;

#if(ENABLE_RADIATION == 1)
    else
        if (calcType == ComputeGridValueOptions::calcLarmorEnergy)
        return UNIT_ENERGY;
#endif
    else
        return 1.0;
}

template<class T_ParticleShape, uint32_t calcType>
HINLINE std::string
ComputeGridValuePerFrame<T_ParticleShape, calcType>::getName() const
{
    if (calcType == ComputeGridValueOptions::calcDensity)
        return "Density";
    else
        if (calcType == ComputeGridValueOptions::calcEnergy)
        return "ParticleEnergy";
    else
        if (calcType == ComputeGridValueOptions::calcEnergyDensity)
        return "EnergyDensity";
    else
        if (calcType == ComputeGridValueOptions::calcCounter)
        return "ParticleCounter";
#if(ENABLE_RADIATION == 1)
    else
        if (calcType == ComputeGridValueOptions::calcLarmorEnergy)
        return "fields_ParticleLarmorEnergy";
#endif
    else
        return "FieldTmp";
}

template<class T_ParticleShape, uint32_t calcType>
template<class FrameType, class TVecSuperCell, class BoxTmp >
DINLINE void
ComputeGridValuePerFrame<T_ParticleShape, calcType>::operator()
(FrameType& frame,
 const int localIdx,
 const TVecSuperCell superCell,
 BoxTmp& tmpBox)
{

    PMACC_AUTO(particle, frame[localIdx]);
    typedef float_X WeightingType;

    const float_X weighting = particle[weighting_];
    const floatD_X pos = particle[position_];
    const float3_X mom = particle[momentum_];
#if(ENABLE_RADIATION == 1)
    const float3_X mom_mt1 = particle[momentumPrev1_];
    const float3_X mom_dt = mom - mom_mt1;
#endif
    const float_X mass = attribute::getMass(weighting,particle);
    const float_X charge = attribute::getCharge(weighting,particle);

    const int particleCellIdx = particle[localCellIdx_];
    const DataSpace<TVecSuperCell::dim> localCell(DataSpaceOperations<TVecSuperCell::dim>::map(superCell,particleCellIdx));

    Gamma<float_X> calcGamma;
    const typename Gamma<float_X>::valueType gamma = calcGamma(mom, mass);
    const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

    const float_X energy = ( gamma <= float_X(GAMMA_THRESH) ) ?
        math::abs2(mom) / ( float_X(2.0) * mass ) :   /* non-relativistic */
        (gamma - float_X(1.0)) * mass * c2;           /* relativistic     */
#if(ENABLE_RADIATION == 1)
    const float_X el_factor = charge * charge
        / (6.0 * PI * EPS0 *
           c2 * c2 * SPEED_OF_LIGHT * mass * mass);
    const float_X energyLarmor = el_factor * math::pow(gamma, 4)
        * (math::abs2(mom_dt) -
           math::abs2(math::cross(mom, mom_dt)));
#endif
    const float_X particleChargeDensity = charge / (CELL_VOLUME);

    /** Shift to the cell the particle belongs to */
    PMACC_AUTO(fieldTmpShiftToParticle, tmpBox.shift(localCell));

    /** loop around local super cell position (regarding shape)
     * \todo take care of non-yee cells
     */
    const DataSpace<simDim> lowMargin(LowerMargin().toRT());
    const DataSpace<simDim> upMargin(UpperMargin().toRT());

    const DataSpace<simDim> marginSpace(upMargin + lowMargin + 1);

    const int numWriteCells = marginSpace.productOfComponents();

    for (int i = 0; i < numWriteCells; ++i)
    {
        /* multidimensionalIndex is only positive: defined range = [0,LowerMargin+UpperMargin]*/
        const DataSpace<simDim> multidimensionalIndex = DataSpaceOperations<simDim>::map(marginSpace, i);
        /* transform coordinate system that it is relative to particle
         * offsetToBaseCell defined range = [-LowerMargin,UpperMargin]
         */
        const DataSpace<simDim> offsetToBaseCell = multidimensionalIndex - lowMargin;
        floatD_X assign;
        for (uint32_t d = 0; d < simDim; ++d)
            assign[d] = AssignmentFunction()(float_X(offsetToBaseCell[d]) - pos[d]);


        /** multiply charge, devide by cell volume and multiply by
         * energy of this particle
         */
        const float_X assignComb = assign.productOfComponents();

        if (calcType == ComputeGridValueOptions::calcDensity)
            atomicAddWrapper(&(fieldTmpShiftToParticle(offsetToBaseCell).x()),
                             assignComb * particleChargeDensity);

        if (calcType == ComputeGridValueOptions::calcEnergy)
            atomicAddWrapper(&(fieldTmpShiftToParticle(offsetToBaseCell).x()),
                             assignComb * energy);

        if (calcType == ComputeGridValueOptions::calcEnergyDensity)
            atomicAddWrapper(&(fieldTmpShiftToParticle(offsetToBaseCell).x()),
                             assignComb * particleChargeDensity * energy);

        if (calcType == ComputeGridValueOptions::calcCounter)
            atomicAddWrapper(&(fieldTmpShiftToParticle(offsetToBaseCell).x()),
                             assignComb * weighting / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);

#if(ENABLE_RADIATION == 1)
        if (calcType == ComputeGridValueOptions::calcLarmorEnergy)
            atomicAddWrapper(&(fieldTmpShiftToParticle(offsetToBaseCell).x()),
                             assignComb * energyLarmor);
#endif


    }
}

} // namespace particleToGrid
} // namespace picongpu
