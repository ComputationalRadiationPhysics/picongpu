/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.hpp"
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
                TVecSuperCell const superCell,
                T_AccFilter& accFilter,
                BoxTmp& tmpBox)
            {
                /* \todo in the future and if useful, the functor can be a parameter */
                T_DerivedAttribute particleAttribute;

                // Only particles passing the filter contribute
                if(accFilter(worker, particle))
                {
                    /* particle attributes: in-cell position and generic, derived attribute */
                    floatD_X const pos = particle[position_];
                    auto const particleAttr = particleAttribute(particle);

                    /** Shift to the cell the particle belongs to
                     * range of particleCell: [DataSpace<simDim>::create(0), TVecSuperCell]
                     */
                    int const particleCellIdx = particle[localCellIdx_];
                    DataSpace<TVecSuperCell::dim> const particleCell
                        = pmacc::math::mapToND(SuperCellSize::toRT(), particleCellIdx);
                    auto fieldTmpShiftToParticle = tmpBox.shift(particleCell);

                    /* loop around the particle's cell (according to shape) */
                    DataSpace<simDim> const lowMargin(LowerMargin().toRT());
                    DataSpace<simDim> const upMargin(UpperMargin().toRT());

                    DataSpace<simDim> const marginSpace(upMargin + lowMargin + 1);

                    int const numWriteCells = marginSpace.productOfComponents();

                    for(int i = 0; i < numWriteCells; ++i)
                    {
                        /** for the current cell i the multi dimensional index currentCell is only positive:
                         * allowed range = [DataSpace<simDim>::create(0), LowerMargin+UpperMargin]
                         */
                        DataSpace<simDim> const currentCell = pmacc::math::mapToND(marginSpace, i);

                        /** calculate the offset between the current cell i with simDim index currentCell
                         * and the cell of the particle (particleCell) in cells
                         */
                        DataSpace<simDim> const offsetParticleCellToCurrentCell = currentCell - lowMargin;

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
                        alpaka::atomicAdd(
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
