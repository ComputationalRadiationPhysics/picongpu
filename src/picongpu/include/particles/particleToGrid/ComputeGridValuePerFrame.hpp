/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
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
#include "pmacc_types.hpp"

#include "math/Vector.hpp"
#include "particles/particleToGrid/ComputeGridValuePerFrame.def"
#include "particles/particleToGrid/derivedAttributes/DerivedAttributes.hpp"

#include "algorithms/Gamma.hpp"

#include <vector>

namespace picongpu
{
namespace particleToGrid
{

template<class T_ParticleShape, class T_DerivedAttribute>
HDINLINE float1_64
ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::getUnit() const
{
    return T_DerivedAttribute().getUnit();
}

template<class T_ParticleShape, class T_DerivedAttribute>
HDINLINE std::vector<float_64>
ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::getUnitDimension() const
{
    return T_DerivedAttribute().getUnitDimension();
}

template<class T_ParticleShape, class T_DerivedAttribute>
HINLINE std::string
ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::getName() const
{
    return T_DerivedAttribute().getName();
}

template<class T_ParticleShape, class T_DerivedAttribute>
template<class FrameType, class TVecSuperCell, class BoxTmp >
DINLINE void
ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::operator()
(FrameType& frame,
 const int localIdx,
 const TVecSuperCell superCell,
 BoxTmp& tmpBox)
{
    /* \todo in the future and if useful, the functor can be a parameter */
    T_DerivedAttribute particleAttribute;

    PMACC_AUTO(particle, frame[localIdx]);

    /* particle attributes: position and generic, derived attribute */
    const floatD_X pos = particle[position_];
    const PMACC_AUTO(particleAttr, particleAttribute( particle ));

    /** Shift to the cell the particle belongs to */
    const int particleCellIdx = particle[localCellIdx_];
    const DataSpace<TVecSuperCell::dim> localCell(
        DataSpaceOperations<TVecSuperCell::dim>::map( superCell, particleCellIdx )
    );
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


        /** multiply charge, divide by cell volume and multiply by
         * energy of this particle
         */
        const float_X assignComb = assign.productOfComponents();

        atomicAddWrapper(&(fieldTmpShiftToParticle(offsetToBaseCell).x()),
                         assignComb * particleAttr);
    }
}

} // namespace particleToGrid
} // namespace picongpu
