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
    uint32_t T_maxNumBins,
    uint32_t T_maxNumNewBins // this may be better names as smth bookkeeping or memory related
>
struct Histogram
{
private:

    constexpr static uint32_t maxNumBins = T_maxNumBins;
    constexpr static uint32_t maxNumNewBins = T_maxNumNewBins;

    //content of bins
    float_X binWeights[ maxNumBins ];
    float_X binDeltaEnergy[ maxNumBins ];
    //location of bins
    uint32_t binIndices[ maxNumBins ];
    // number of bins occupied
    uint32_t numBins;

    //new Bins form current Iteration
    float_X  newBinsWeights[ maxNumNewBins ];
    uint32_t newBinsIndices[ maxNumNewBins ];
    uint32_t numNewBins;

    //
    float_X const binWidth;

public:

    DINLINE Histogram( float_X const binWidth ):
        binWidth( binWidth )
    {
        this->numBins = 0u;
        this->numNewBins = 0u;

        // For debug purposes this is okay
        // Afterwards this code should be removed as we are
        // filling memory we are never touching (if everything works)
        for( uint32_t i = 0u; i < maxNumBins; i++ )
        {
            this->binWeights[i] = 0.;
            this->binDeltaEnergies = 0.;
            this->binIndices[i] = 0u;
        }

        for( uint32_t i = 0u; i < numThreads; i++)
        {
            this->newBinsIndices[i] = 0;
            this->newBinsWeights[i] = 0;
        }

    }

    // Return index in the binIndices array when present
    // or maxNumBin when not present
    DINLINE uint32_t findBin(
        uint32_t binIndex,
        uint32_t startIndex = 0u
    ) const
    {
        for(uint32_t i = startIndex; i < numBins; i++)
        {
            if( this->binIndices[i] == binIndex )
            {
                return i;
            }
        }
        return maxNumBin;
    }

    DINLINE bool hasBin( uint32_t binIndex ) const
    {
        auto const index = findBin( binIndex );
        return index < maxNumBins;
    }

    DINLINE uint32_t getBinIndex( float_X energy ) const
    {
        //standard fixed bin width
        return static_cast< uint32_t >( energy/binWidth );
    }

    /// perhaps it's better if this function takes already
    /// x (energy) and weight and the caller computes those
    template<
        typename T_particle
    >
    DINLINE void binObject( T_particle particle)
    {
        float_X const m = particle[ massRatio_ ] * SI::BASE_MASS;     //Unit: kg
        constexpr auto c = SPEED_OF_LIGHT;                   //Unit: m/s

        float3_X vectorP = this->getIndex( particle[ momentum_ ] );
        // we probably have a math function for ||p||^2
        float_X pSquared = vectorP[0]*vectorP[0] +
            vectorP[1]*vectorP[1] +
            vectorP[2]*vectorP[2];                      //unit:kg*m/s

        //calculate Energy
        float_X energy = static_cast< float_X >(
            std::sqrt(                                  //unit: kg*m^2/s^2 = Nm
                m*m * c*c*c*c + pSquared * c*c
                )
            );

        // compute bin index
        uint32_t const binIndex = this->getBinIndex( energy );

        //search for bin in existing bins
        auto const index = findBin(binIndex);

        // check for range of index
        // TODO: this is not compatible to alpaka
        if( index < 0 or index > maxNumBins)
        {
            throw new std::range_error(
                "bad Histogram Storage Index upon binObject call"
                );
        }

        // If the bin was already there, we need to atomically increase
        // the value, as another thread may contribute to the same bin
        if( index < maxNumBin )
        {
            /// probably needs acc
            cupla::atomicAdd(
                &(this->binValues[ index ]),
                particle[ weighting_ ]
            );
        }
        else
        {
            // Otherwise we add it to a collection of new bins
            // Note: in current dev the namespace is different in cupla
            auto newBinIdx = cupla::atomicInc( numNewBins );
            if( newBinIdx < maxNumNewBins )
            {
                newBinsWeights[ newBinIdx ] = particle[ weighting_ ];
                newBinsIndices[ newBinIdx ] = binIndex;
            }
            else
            {
                /// If we are here, there were more new bins since the last update
                // call than memory allocated for them
                // Normally, this should not happen
            }
        }
    }

    // This is to move new bins to the main collection of bins
    // Should be called periodically so that we don't run out of memory for
    // new bins
    // Must be called sequentially
    DINLINE void updateWithNewBins()
    {
        uint32_t const numBinsBeforeUpdate = numBins;
        for (uint32_t i = 0u; i < this->numNewBins; i++ )
        {
            // New bins were definitely not present before
            // But several new bins can be the same actual bin
            // So we search in the newly added part only
            auto auto const index = findBin(
                this->newBinsIndices[i],
                numBinsBeforeUpdate
            );

            // If this bin was already added
            if( index < maxNumBin )
                this->binWeights[ index ] += this->newBinsWeights[ i ];
            else
            {
                if( this->numBins < this->maxNumBins )
                {
                    this->binWeights[ this->numBins ] = this->newBinsWeights[i];
                    this->binIndices[ this->numBins ] = this->newBinsIndices[i];
                    this->numBins++;
                }
                // else we ran out of memory, do nothing
            }
        }
        this->numNewBins = 0u;
    }
}

} // namespace histogram2
} // namespace electronDistribution
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
