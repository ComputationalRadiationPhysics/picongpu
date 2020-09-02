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

/** @file This file implements a adaptive Histogram starting from argument 0
 */

#pragma once


#include "picongpu/simulation_defines.hpp"
#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <utility>
#include "picongpu/traits/attribute/GetMass.hpp"


namespace picongpu
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
        uint32_t T_maxNumNewBins
    >
    struct AdaptiveHistogram
    {
    private: // TODO: chnage all calls to new getter methods
        constexpr static uint32_t maxNumBins = T_maxNumBins;
        constexpr static uint32_t maxNumNewBins = T_maxNumNewBins;

        // content of bins
        // two data fields
        // weight of particles
        float_X binWeights[ maxNumBins ];
        // chnage in particle Energy
        float_X binDeltaEnergy[ maxNumBins ];

        //x of bins, by global histogram index
        uint32_t binIndices[ maxNumBins ];
        // number of bins occupied
        uint32_t numBins;

        // new bins since last call to update method
        uint32_t newBinsIndices[ maxNumNewBins ];
        float_X  newBinsWeights[ maxNumNewBins ];
        // number of entries in new bins
        uint32_t numNewBins;

        // boundary of some bin in the histogram
        float_X lastBoundary;
        // Index of the bin whose left boundary is lastBoundary
        uint32_t lastBinIndex;

        // target value of relative Error of histogram Bin,
        // bin width choosen such that relativeError as close to target
        // TODO: make template parameter
        float_X relativeErrorTarget;

        // defines initial global grid
        float_X initialGridWidth;

    public:
        // Has to be called by one thread before any other method
        // of this object to correctly initialise the histogram
        // @param relativeErrorTarget ... should be >0,
        //      maximum relative Error of Histogram Bin
        DINLINE void init(
            float_X relativeErrorTarget,
            float_X initialGridWidth
            )
        {
            // init histogram empty
            this->numBins = 0u;
            this->numNewBins = 0u;

            // init with 0 as reference point
            this->lastBoundary = 0._X;
            this->lastBinIndex = 0u;

            // init adaptive bin width algorithm parameters
            this->relativeErrorTarget = relativeErrorTarget;
            this->initialGridWidth = initialGridWidth;

            // TODO: make this debug mode only
            // For debug purposes this is okay
            // Afterwards this code should be removed as we are
            // filling memory we are never touching (if everything works)

            // start of debug init
            for( uint32_t i = 0u; i < maxNumBins; i++ )
            {
                this->binWeights[i] = 0.;
                this->binDeltaEnergy[i] = 0._X;
                this->binIndices[i] = 0u;
            }

            for( uint32_t i = 0u; i < maxNumNewBins; i++)
            {
                this->newBinsIndices[i] = 0;
                this->newBinsWeights[i] = 0._X;
            }
            // end of debug init
        }

        // Tries to find binIndex in the collection and return the collection index.
        // Returns index in the binIndices array when present
        // or maxNumBin when not present,
        // maxNumBin is never a valid index of the collection
        // uses stupid linear search for now
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
            return maxNumBins;
        }

        // checks whether Bin exists
        DINLINE bool hasBin( uint32_t binIndex ) const
        {
            auto const index = findBin( binIndex );
            return index < maxNumBins;
        }

        DINLINE static constexpr uint32_t getMaxNumberBins ()
        {
            return AdaptiveHistogram::maxNumBins;
        }

        DINLINE static constexpr uint32_t getMaxNumberNewBins ()
        {
            return AdaptiveHistogram::maxNumNewBins;
        }

        // return center of Bin
        DINLINE static float_X centerBin(
            bool directionPositive,
            float_X Boundary,
            float_X binWidth
            )
        {
            if ( directionPositive )
                return Boundary + binWidth/2._X;
            else
                return Boundary - binWidth/2._X;
        }

        // is x in Bin 
        //  - directionPositive == true:    [boundary,           boundary + binWidth)
        //  - directionPositive == false:   [boundary - binWidth,boundary)
        DINLINE static bool inBin(
            bool directionPositive,
            float_X boundary,
            float_X binWidth,
            float_X x
            )
        {
            if ( directionPositive )
                return ( x >= boundary ) && ( x < boundary + binWidth );
            else
                return ( x >= boundary - binWidth) && ( x < boundary );
        }

        // relative error function used
        DINLINE static float_X relativeErrorFunction(
            float_X binWidth,
            float_X centralValue
            )
        {
            return 0._X;
        }

        DINLINE float_X getBinWidth(
            const bool directionPositive,
            const float_X boundary,
            float_X currentBinWidth
            ) const
        {
            // is initial binWidth below the Target Value?
            bool isBelowTarget = (
                this->relativeErrorTarget >= relativeErrorFunction(
                    currentBinWidth,
                    AdaptiveHistogram::centerBin(
                        directionPositive,
                        boundary,
                        currentBinWidth
                        )
                    )
                );


            if( isBelowTarget )
            {
                // increase until no longer below
                while ( isBelowTarget )
                {
                    // try higher binWidth
                    currentBinWidth *= 2._X;

                    // until no longer below Target
                    isBelowTarget = (
                        this->relativeErrorTarget > relativeErrorFunction(
                            currentBinWidth,
                            AdaptiveHistogram::centralBin(
                                directionPositive,
                                currentBinWidth,
                                boundary
                                )
                            )
                        );
                }

                // last i-th try was not below target,
                // but (i-1)-th was still below
                // -> reset to value i-1
                currentBinWidth /= 2._X;
            }
            else
            {
                // decrease until below targetf or the first time
                while ( !isBelowTarget )
                {
                    // try lower binWidth
                    currentBinWidth /= 2._X;

                    // until first time below Target
                    isBelowTarget = (
                        this->relativeErrorTarget > relativeErrorFunction(
                            currentBinWidth,
                            AdaptiveHistogram::centralBin(
                                directionPositive,
                                currentBinWidth,
                                boundary
                                )
                            )
                        );
                }
                // no need to reset to value before
                // since this was first value that was below target
            }

            return currentBinWidth;
        }

        // Returns the bin index, number identifying every bin possible in the
        // histogram
        // unrelated to collection index, can be larger than maxNumBins
        // actual adaptive part of histogram
        // @param x ... where object is located that we want to bin
        // does not change last binIndex
        DINLINE  uint32_t getBinIndex( float_X x ) const
        {

            // wether x is in positive direction with regards to last known
            // Boundary
            bool directionPositive = ( x >= this->lastBoundary );

            // init currentBinWidth with initial grid
            float_X currentBinWidth = this->initialGridWidth;

            // start from prevoius point to reduce seek times
            float_X boundary = this->lastBoundary;
            uint32_t index = this->lastBinIndex;

            bool inBin = false;

            while ( !inBin )
            {

                // get currentBinWidth
                currentBinWidth = getBinWidth(
                    directionPositive,
                    boundary,
                    currentBinWidth
                    );

                inBin = AdaptiveHistogram::inBin(
                        directionPositive,
                        boundary,
                        currentBinWidth,
                        x
                        );

                // check wether x is in Bin
                if ( inBin )
                    // yes case, => current index is correct index
                    return index;
                else
                {
                    if ( directionPositive )
                    {
                        index += 1;
                        boundary += currentBinWidth;
                    }
                    else
                    {
                        // check for underflow
                        if ( index > 0 )
                        {
                            index -= 1;
                            boundary -= currentBinWidth;
                        }
                        else
                        // add to 0-th bin instead
                        // TODO: allow negative indices as well
                        // useful for phase space histograms where the argument
                        // (position vector component or impulse component) can be <0
                        // not necessary for energy histogram, since E>=0
                            return 0u;
                    }
                }
            }
        }


        // add for Argument x, weight to the bin
        template< typename T_Acc >
        DINLINE void binObject(
            T_Acc const & acc,
            float_X const x,
            float_X const weight
        )
        {
            // compute global bin index
            uint32_t const binIndex = this->getBinIndex( x );

            //search for bin in collection of existing bins
            auto const index = findBin( binIndex );

            // If the bin was already there, we need to atomically increase
            // the value, as another thread may contribute to the same bin
            if( index < maxNumBins ) // bin already exists
            {
                cupla::atomicAdd(
                    acc,
                    &(this->binWeights[ index ]),
                    weight
                );
            }
            else
            {
                // Otherwise we add it to a collection of new bins
                // Note: in current dev the namespace is different in cupla
                // get Index where to deposit it by atomic add to numNewBins
                // this assures that the same index is not used twice
                auto newBinIdx = cupla::atomicAdd< alpaka::hierarchy::Threads >(
                    acc,
                    &numNewBins,
                    1u
                );
                if( newBinIdx < maxNumNewBins )
                {
                    newBinsWeights[ newBinIdx ] = weight;
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

        // This moves new bins to the main collection of bins
        // Should be called periodically so that we don't run out of memory for
        // new bins
        // choose maxNumNewBins to fit period and number threads and particle frame
        // size
        // MUST be called sequentially
        DINLINE void updateWithNewBins()
        {
            uint32_t const numBinsBeforeUpdate = numBins;
            for (uint32_t i = 0u; i < this->numNewBins; i++ )
            {
                // New bins were definitely not present before
                // But several new bins can be the same actual bin
                // So we search in the newly added part only
                auto const index = findBin(
                    this->newBinsIndices[i],
                    numBinsBeforeUpdate
                );

                // If this bin was already added
                if( index < maxNumBins )
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
    };

} // namespace histogram2
} // namespace electronDistribution
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
