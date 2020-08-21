/* Copyright 2020 Brian Marre
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

/** @file description basic experimental implementation of Histogram
 */

#pragma once

#include <cmath>
#inlcude <stdexcept>

#include <picongpu/traits/attribute/GetMass.hpp>
#include <picongpu/traits/attribute/GetCharge.hpp>



namespace namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
namespace electronDistribution
{
namespace histogram2
{

template<
    uint32_t maxNBins,
    uint32_t numThreads,
    typename T_Particle,
    float_X binWidth
>
struct Histogram
{
private:
    //content of bins
    float_X binWeights[ maxNumBins ];
    float_X binDeltaEnergy[ maxNumBins ];
    //location of bins
    uint32_t binIndices[ maxNumBins ];
    // number of bins occupied
    uint32_t numBins;

    //new Bins form current Iteration
    float_X  newBinsWeights[ numThreads ];
    uint32_t newBinsIndices[ numThreads ];

public:
    AdaptiveHistogram()
    {
        this->numBins = 0u;

        for( uint32_t i = 0u, i < maxNumBins, i++ )
        {
            this->binWeights[i] = 0.;
            this->binDeltaEnergies = 0.;
            this->binIndices[i] = 0u;
        }

        for( uint32_t i = 0u, i < numThreads, i++)
        {
            this->newBinsIndices[i] = 0;
            this->newBinsWeights[i] = 0;
        }

    }

    uint32_t findBin( uint32_t binIndex )
    {
        for(uint32_t i = 0, i < maxNumBins, i++)
        {
            if( this->binIndices[i] == binIndex )
            {
                return i;
            }
        }
        return maxNumBin;
    }

    bool hasBin(uint32_t binIndex)
    {
        for(uint32_t i = 0, i < maxNumBins, i++)
        {
            if( this->binIndices[i] == binIndex )
            {
                return true;
            }
        }
        return false;
    }

    uint32_t getBinIndex( float_X energy )
    {
        //standard fixed bin width
        return static_cast< uint32_t >( energy/binWidth );
    }

    template<
        T_particle
        >
    void binObject( T_particle particle, uint32_t threadIndex)
    {
        uint32_t binIndex;

        float_X m;
        float_X c;
        float_X pSquared;
        float_X energy;
        float3_X vectorP;

        m = particle[ massRatio_ ] * SI::BASE_MASS;     //Unit: kg
        constexpr c = SPEED_OF_LIGHT;                   //Unit: m/s

        vectorP = this->getIndex( particle[ momentum_ ] );
        pSquared = vectorP[0]*vectorP[0] +
            vectorP[1]*vectorP[1] +
            vectorP[2]*vectorP[2];                      //unit:kg*m/s

        //calculate Energy
        energy = static_cast< float_X >(
            std::sqrt(                                  //unit: kg*m^2/s^2 = Nm
                m*m * c*c*c*c + pSquared * c*c
                )
            );

        //find Bin
        binIndex = this->getBinIndex( Energy );

        //search for bin in existing bins
        index = findBin(binIndex);

        // check for range of index
        if( index < 0 or index > maxNumBins)
        {
            throw new std::range_error(
                "bad Histogram Storage Index upon binObject call"
                );
        }

        //case bin not found
        if ( index == maxNumBin )
        {
            if ( this->numBins == maxNumBins )
            {
                throw new std::overflow_error("maximum number of bins exceeded on"
                "binning");
            }

            newBinsWeights[ threadIndex ] = weight;
            newBinsIndices[ threadIndex ] = binIndex;
        }
        else
        {
            this->binValues[ index ] += weight;
        }
    }

    void updateWithNewBins()
    {
        for (uint32_t i = 0, i < numThreads, i++ )
        {
            if( this->numBins < maxNumBins-1 and this->newBinsWeights[i] == 0 )
            {
                this->binValues[ this->numBins ] = this->newBinsWeights[i];
                this->binIndices[ this->numBins ] = this->newBinsIndices[i];
                this->numBins++;
                this->newBinsWeights[i] = 0;
                this->newBinsIndices[i] = 0;
            }
            else
            {
                throw new std:overflow_error("maxNumBins exeded on update with new");
            }
        }
    }
}

} // namespace histogram2
} // namespace electronDistribution
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
