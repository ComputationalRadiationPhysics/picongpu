/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.hpp"

#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

#include <vector>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            template<class T_ParticleShape, class T_DerivedAttribute>
            HDINLINE float1_64 ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::getUnit() const
            {
                return T_DerivedAttribute().getUnit();
            }

            template<class T_ParticleShape, class T_DerivedAttribute>
            HINLINE std::vector<float_64> ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::
                getUnitDimension() const
            {
                return T_DerivedAttribute().getUnitDimension();
            }

            template<class T_ParticleShape, class T_DerivedAttribute>
            HINLINE std::string ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::getName()
            {
                return T_DerivedAttribute::getName();
            }

            template<class T_ParticleShape, class T_DerivedAttribute>
            template<
                typename T_Particle,
                typename TVecSuperCell,
                typename BoxTmp,
                typename T_Worker,
                typename T_AccFilter>
            DINLINE void ComputeGridValuePerFrame<T_ParticleShape, T_DerivedAttribute>::operator()(
                T_Worker const& worker,
                T_Particle& particle,
                const TVecSuperCell superCell,
                T_AccFilter& accFilter,
                BoxTmp& tmpBox)
            {
                /* \todo in the future and if useful, the functor can be a parameter */
                T_DerivedAttribute particleAttribute;

                // Only particles passing the filter contribute
                if(accFilter(worker, particle))
                {
                    /* particle attributes: in-cell position and generic, derived attribute */
                    const floatD_X pos = particle[position_];
                    const auto particleAttr = particleAttribute(particle);

                    /** Shift to the cell the particle belongs to
                     * range of particleCell: [DataSpace<simDim>::create(0), TVecSuperCell]
                     */
                    const int particleCellIdx = particle[localCellIdx_];
                    const DataSpace<TVecSuperCell::dim> particleCell(
                        DataSpaceOperations<TVecSuperCell::dim>::map(superCell, particleCellIdx));
                    auto fieldTmpShiftToParticle = tmpBox.shift(particleCell);

                    /* loop around the particle's cell (according to shape) */
                    const DataSpace<simDim> lowMargin(LowerMargin().toRT());
                    const DataSpace<simDim> upMargin(UpperMargin().toRT());

                    const DataSpace<simDim> marginSpace(upMargin + lowMargin + 1);

                    const int numWriteCells = marginSpace.productOfComponents();

                    for(int i = 0; i < numWriteCells; ++i)
                    {
                        /** for the current cell i the multi dimensional index currentCell is only positive:
                         * allowed range = [DataSpace<simDim>::create(0), LowerMargin+UpperMargin]
                         */
                        const DataSpace<simDim> currentCell = DataSpaceOperations<simDim>::map(marginSpace, i);

                        /** calculate the offset between the current cell i with simDim index currentCell
                         * and the cell of the particle (particleCell) in cells
                         */
                        const DataSpace<simDim> offsetParticleCellToCurrentCell = currentCell - lowMargin;

                        /** assign particle contribution component-wise to the lower left corner of
                         * the cell i
                         * \todo take care of non-yee cells
                         */
                        float_X assign(1.0);
                        for(uint32_t d = 0; d < simDim; ++d)
                            assign *= AssignmentFunction()(float_X(offsetParticleCellToCurrentCell[d]) - pos[d]);

                        /** add contribution of the particle times the generic attribute
                         * to cell i
                         * note: the .x() is used because FieldTmp is a scalar field with only
                         * one "x" component
                         */
                        cupla::atomicAdd(
                            worker.getAcc(),
                            &(fieldTmpShiftToParticle(offsetParticleCellToCurrentCell).x()),
                            assign * particleAttr,
                            ::alpaka::hierarchy::Threads{});
                    }
                }
            }

        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
