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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once


#include "simulation_defines.hpp"
#include "particles/startPosition/MacroParticleCfg.hpp"
#include "particles/startPosition/IFunctor.def"

namespace picongpu
{
namespace particles
{
namespace startPosition
{

template<typename T_ParamClass>
struct QuietImpl
{

    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        typedef QuietImpl<ParamClass> type;
    };

    HINLINE QuietImpl(uint32_t): numParInCell(ParamClass::numParticlesPerDimension::toRT())
    {
    }

    DINLINE void init(const DataSpace<simDim>& totalCellOffset)
    {

    }

    /** Distributes the initial particles lattice-like within the cell.
     *
     * @param rng a reference to an initialized, UNIFORM random number generator
     * @param curParticle the number of this particle: [0, totalNumParsPerCell-1]
     * @return float3_X with components between [0.0, 1.0)
     */
    DINLINE floatD_X operator()(const uint32_t curParticle)
    {
        // spacing between particles in each direction in the cell
        DataSpace<simDim> numParDirection(numParInCell);
        floatD_X spacing;
        for (uint32_t i = 0; i < simDim; ++i)
            spacing[i] = (float_X(1.0) / float_X(numParDirection[i]));

        // coordinate in the local in-cell lattice
        //   x = [0, numParsPerCell_X-1]
        //   y = [0, numParsPerCell_Y-1]
        //   z = [0, numParsPerCell_Z-1]
        DataSpace<simDim> inCellCoordinate = DataSpaceOperations<simDim>::map(numParDirection, curParticle);


        return floatD_X(precisionCast<float_X>(inCellCoordinate) * spacing + spacing * float_X(0.5));
    }

    /** If the particles to initialize (numParsPerCell) end up with a
     *  related particle weighting (macroWeighting) below MIN_WEIGHTING,
     *  reduce the number of particles if possible to satisfy this condition.
     *
     * @param numParsPerCell the intendet number of particles for this cell
     * @param realElPerCell  the number of real electrons in this cell
     * @return macroWeighting the intended weighting per macro particle
     */
    DINLINE MacroParticleCfg mapRealToMacroParticle(const float_X realElPerCell)
    {
        float_X macroWeighting = float_X(0.0);
        uint32_t numParsPerCell=numParInCell.productOfComponents();

        if (numParsPerCell > 0)
            macroWeighting = realElPerCell / float_X(numParsPerCell);

        while (macroWeighting < MIN_WEIGHTING &&
               numParsPerCell > 0)
        {
            /* decrement component with greatest value*/
            uint32_t max_component = 0;
            for (uint32_t i = 1; i < simDim; ++i)
            {
                if (numParInCell[i] > numParInCell[max_component])
                    max_component = i;
            }
            numParInCell[max_component] -= 1;

            numParsPerCell = numParInCell.productOfComponents();

            if (numParsPerCell > 0)
                macroWeighting = realElPerCell / float_X(numParsPerCell);
            else
                macroWeighting = float_X(0.0);
        }

        MacroParticleCfg macroParCfg;
        macroParCfg.weighting = macroWeighting;
        macroParCfg.numParticlesPerCell = numParsPerCell;

        return macroParCfg;
    }

protected:

    DataSpace<simDim> numParInCell;
};
} //namespace particlesStartPosition
} //namespace particles
} //namespace picongpu
