/* Copyright 2020-2023 Brian Marre
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

/** @file This file implements a fixed bin width histogram starting from Argument 0.
 * This is done as an unordered collection of bins indices and their corresponding
 * values, this implementation was done for testing purposes for the adaptive
 * histogram.
 */

#pragma once


#include "picongpu/simulation_defines.hpp"

#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>


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
                    template<uint32_t T_maxNumBins, uint32_t T_maxNumNewBins>
                    struct Histogram
                    {
                    private: /// @todo : change all calls to new getter methods
                        constexpr static uint32_t maxNumBins = T_maxNumBins;
                        constexpr static uint32_t maxNumNewBins = T_maxNumNewBins;

                        // content of bins
                        float_X binWeights[maxNumBins];
                        float_X binDeltaEnergy[maxNumBins];
                        // location of bins
                        uint32_t binIndices[maxNumBins];
                        // number of bins occupied
                        uint32_t numBins;

                        // new Bins form current Iteration
                        float_X newBinsWeights[maxNumNewBins];
                        uint32_t newBinsIndices[maxNumNewBins];
                        uint32_t numNewBins;

                        float_X binWidth;

                    public:
                        DINLINE static constexpr uint32_t getMaxNumberBins()
                        {
                            return Histogram::maxNumBins;
                        }

                        DINLINE static constexpr uint32_t getMaxNumberNewBins()
                        {
                            return Histogram::maxNumNewBins;
                        }

                        // Has to be called by one thread before any other method
                        // of this object to correctly initialise the histogram
                        DINLINE void init(float_X const binWidth)
                        {
                            this->binWidth = binWidth;
                            this->numBins = 0u;
                            this->numNewBins = 0u;

                            /// @todo : make this debug mode only
                            /// For debug purposes this is okay
                            /// Afterwards this code should be removed as we are
                            /// filling memory we are never touching (if everything works)
                            for(uint32_t i = 0u; i < maxNumBins; i++)
                            {
                                this->binWeights[i] = 0.;
                                this->binDeltaEnergy[i] = 0.;
                                this->binIndices[i] = 0u;
                            }

                            for(uint32_t i = 0u; i < maxNumNewBins; i++)
                            {
                                this->newBinsIndices[i] = 0;
                                this->newBinsWeights[i] = 0;
                            }
                        }

                        /** Tries to find binIndex in the collection and return the collection index.
                         * Returns index in the binIndices array when present or maxNumBin
                         * when not present, maxNumBin is never a valid index of the collection
                         * stupid linear search for now, @todo
                         */
                        DINLINE uint32_t findBin(uint32_t binIndex, uint32_t startIndex = 0u) const
                        {
                            for(uint32_t i = startIndex; i < numBins; i++)
                            {
                                if(this->binIndices[i] == binIndex)
                                {
                                    return i;
                                }
                            }
                            return maxNumBins;
                        }

                        // checks wether Bin exists, exists only for encapsulation
                        DINLINE bool hasBin(uint32_t binIndex) const
                        {
                            auto const index = findBin(binIndex);
                            return index < maxNumBins;
                        }

                        // Returns the bin index, number identifying every bin possible in the
                        // histogram
                        // unrelated to collection index, can be larger than maxNumBins
                        DINLINE uint32_t getBinIndex(float_X energy) const
                        {
                            // standard fixed bin width
                            return static_cast<uint32_t>(energy / binWidth);
                        }

                        // add for Argument x, weight to the bin
                        template<typename T_Worker>
                        DINLINE void binObject(T_Worker const& worker, float_X const x, float_X const weight)
                        {
                            // compute global bin index
                            uint32_t const binIndex = this->getBinIndex(x);

                            // search for bin in collection of existing bins
                            auto const index = findBin(binIndex);

                            // If the bin was already there, we need to atomically increase
                            // the value, as another thread may contribute to the same bin
                            if(index < maxNumBins) // bin already exists
                            {
                                alpaka::atomicAdd(worker.getAcc(), &(this->binWeights[index]), weight);
                            }
                            else
                            {
                                // Otherwise we add it to a collection of new bins
                                auto newBinIdx = alpaka::atomicAdd(
                                    worker.getAcc(),
                                    &numNewBins,
                                    1u,
                                    alpaka::hierarchy::Threads{});
                                if(newBinIdx < maxNumNewBins)
                                {
                                    newBinsWeights[newBinIdx] = weight;
                                    newBinsIndices[newBinIdx] = binIndex;
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
                            for(uint32_t i = 0u; i < this->numNewBins; i++)
                            {
                                // New bins were definitely not present before
                                // But several new bins can be the same actual bin
                                // So we search in the newly added part only
                                auto const index = findBin(this->newBinsIndices[i], numBinsBeforeUpdate);

                                // If this bin was already added
                                if(index < maxNumBins)
                                    this->binWeights[index] += this->newBinsWeights[i];
                                else
                                {
                                    if(this->numBins < this->maxNumBins)
                                    {
                                        this->binWeights[this->numBins] = this->newBinsWeights[i];
                                        this->binIndices[this->numBins] = this->newBinsIndices[i];
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
