/* Copyright 2020-2021 Brian Marre
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
 *
 * basic idea: 1-dimensional infinite histogram, with sparsely populated bins
 * binWidths are not uniform, but indirectly definded by a target value of a
 * relative binning width dependent error.
 *
 * only populated bins are stored in memory.
 *
 * TemplateParameters:
 * -------------------
 *  @tparam T_AtomicDataBox ... type of data container for atomic input data
 *  @tparam T_maxNumberBins ... maximum number of bins of the histogram
 *  @tparam T_maxNumNewBins ... maximum number of new bins before updateWithNewBins
 *                                must be called by one thread
 *                                BEWARE: causes invalid memory access if set incorrectly
 *
 * private members:
 * ----------------
 *  float_X[T_maxNumberBins] binWeights ... histogram bin values
 *                                          weights of particles in bin in supercell
 *  float_X{T_maxNumberBins] binDeltaEnergy  ... secondary histogram value
 *                                               energy change in this time step in this histogram bin
 *  float_X{T_maxNumberBins] binLeftBoundary ... left boundary of bin in argument space
 *
 * storage of bins added since last call to updateWithNewBins
 * float_X[T_maxNumNewBins] newBinsWeights      ... see regular
 * float_X[T_maxNumNewBins] newBinsLeftBoundary ... -||-
 *  NOTE: no <newBinsDeltaEnergy> necessary, since all electrons are already binned for this step
 *
 *  float_x relativeErrorTarget   ... relative error target used to choose bin size
 *  T_RelativeError relativeError ... relative error functor
 *  float_X initialgridWidth      ... intial binWidth used
 *
 *  uint numBins    ... number of bins occupied
 *  uint numNewBins ... number of newBins occupied,
 *
 *  float_X lastLeftBoundary ... last left boundary of a bin used
 *
 * private methods:
 * ---------------
 *  uint findBin(float_X leftBoundary, uint startIndex=0) ... tries to finds leftBoundary in bin collection
 *  bool hasBin(float_X leftBoundary, uint startIndex=0)  ... checks wether bin exists
 *  float_X centerBin(bool directionPositive, float_X Boundary, float_X binWidth ) ...
 *      return center of Bin,
 *      direction positive indicates wether the leftBoundary(true) or the right Boundary(false) is given as argument
 *  bool inBin(bool directionPositive, float_X boundary, float_X binWidth, float_X x) ...
 *      is x in the given Bin?
 *
 * public methods:
 * ---------------
 *  void init(float_X relativeErrorTarget >0 , float_X initialGridWidth >0 , T_RelativeError& relativeError) ...
 *      init method for adaptive histogram, must be called by one thread once before use
 *      BEWARE: assumptions for input parameters
 *
 *  uint getMaxNumberBins()     ... returns corresponding template parameter value
 *  uint getMaxNUmberNewBins()  ... -||-
 *  float_X getInitialGridWidth()
 *
 *  uint getNumBins() ... returns current number of occupied bins
 *  float_X getEnergyBin(T_Acc& acc, uint index, T_AtomicDataBox atomicDataBox) ...
 *      returns central energy of bin with given collection index if it exists, otherwise returns 0._X
 *  float_X getLeftBoundaryBin(uint index) ... return leftBoundary of bin with given collection index
 *  float_X getWeightBin(uint index) ... returns weight of Bin with given collection index
 *      BEWARE: does not check if the bin is occupied, or exists at all,
 *          leave alone unless you know what you are doing
 *  float_X getDeltaEnergyBin(uint index) ... same as above for delta energy bin, beware also applies
 *  uint getBinIndex(T_Acc& acc, float_X energy, T_AtomicDataBox atomicDataBox) ...
 *      returns collection index of existing bin containing this energy, or maxNumBins otherwise
 *  float_X getBinWidth(T_Acc& acc, bool directionPositive, float_X boundary, T_AtomicDataBox atomicDataBox)
 *      return binWidth of the specifed bin, for more see in code function documentation
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <alpaka/alpaka.hpp>

#include <utility>


// debug only
#include <cmath>
#include <iostream>

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
                    template<uint32_t T_maxNumBins, uint32_t T_maxNumNewBins, typename T_AtomicDataBox>
                    struct AdaptiveHistogram
                    {
                    private:
                        //{ members
                        constexpr static uint32_t maxNumBins = T_maxNumBins;
                        constexpr static uint32_t maxNumNewBins = T_maxNumNewBins;

                        //{ content of bins, three data fields
                        // 1. weight of particles
                        float_X binWeights[maxNumBins];
                        // 2. accumulated change in particle Energy in this bin,
                        float_X binDeltaEnergy[maxNumBins];
                        // 3. accumulated weight added to this bin due
                        float_X binDeltaWeight[maxNumBins];
                        //}

                        // location of bins
                        float_X binLeftBoundary[maxNumBins];

                        //{ administration data
                        // number of bins occupied, <= T_maxNumBins
                        uint16_t numBins;

                        // new bins since last call to update method
                        float_X newBinsLeftBoundary[maxNumNewBins];
                        float_X newBinsWeights[maxNumNewBins];
                        // number of entries in new bins
                        uint32_t numNewBins;
                        //}

                        // reference point of histogram, always boundary of a bin in the histogram
                        // in current implementation not used
                        float_X lastLeftBoundary; // unit: Argument

                        // target value of relative Error of a histogram bin,
                        // bin width choosen such that relativeError is close to target
                        float_X relativeErrorTarget; // unit: varies


                    public:
                        // defines initial global grid
                        float_X initialGridWidth; // unit: Argument
                        //}

                    private:
                        // TODO: replace linear search, by ordering bins
                        /** Tries to find binLeftBoundary in the collection,
                         * starting at the given collection index and return the collection index.
                         *
                         * Returns index in the binIndices array when present or maxNumBin when not present
                         *
                         * maxNumBin is never a valid index of the collection
                         */
                        DINLINE uint16_t findBin(
                            float_X leftBoundary, // unit: Argument
                            uint16_t startIndex = 0u) const
                        {
                            for(uint16_t i = startIndex; i < numBins; i++)
                            {
                                if(this->binLeftBoundary[i] == leftBoundary)
                                {
                                    return i;
                                }
                            }
                            // case bin not found
                            return maxNumBins;
                        }

                        // checks whether Bin exists
                        DINLINE bool hasBin(float_X leftBoundary) const
                        {
                            auto const index = findBin(leftBoundary);
                            return (index < maxNumBins);
                        }

                        // return center of Bin
                        DINLINE static float_X centerBin(
                            bool const directionPositive,
                            float_X const Boundary,
                            float_X const binWidth)
                        {
                            if(directionPositive)
                                return Boundary + binWidth / 2._X;
                            else
                                return Boundary - binWidth / 2._X;
                        }

                        /** =^= is x in Bin?
                         *  case directionPositive == true:    [boundary, boundary + binWidth)
                         *  case directionPositive == false:   [boundary - binWidth,boundary)
                         */
                        DINLINE static bool inBin(
                            bool directionPositive,
                            float_X boundary,
                            float_X binWidth,
                            float_X x)
                        {
                            if(directionPositive)
                                return (x >= boundary) && (x < boundary + binWidth);
                            else
                                return (x >= boundary - binWidth) && (x < boundary);
                        }

                    public:
                        /** Has to be called by one thread once before any other method
                         * of this object to correctly initialise the histogram.
                         *
                         * @param relativeErrorTarget ... should be >0,
                         *      maximum relative Error of Histogram Bin
                         * @param initialGridWidth ... should be >0,
                         *      starting binwidth of algorithm
                         * @param relativeError ... relative should be monoton rising with rising binWidth
                         */
                        DINLINE void init(float_X relativeErrorTarget, float_X initialGridWidth)
                        {
                            // debug only
                            /*printf(
                                "        initialGridWidth_INIT_IN %f, relativeErrorTarget_INIT_IN %f\n",
                                initialGridWidth,
                                relativeErrorTarget);
                            */

                            // init histogram empty
                            this->numBins = 0u;
                            this->numNewBins = 0u;

                            // init with 0 as reference point
                            this->lastLeftBoundary = 0._X;

                            // init adaptive bin width algorithm parameters
                            this->relativeErrorTarget = relativeErrorTarget;
                            this->initialGridWidth = initialGridWidth;

                            // TODO: make this debug mode only
                            // since we are filling memory we are never touching (if everything works)

                            //{ start of debug init
                            for(uint16_t i = 0u; i < maxNumBins; i++)
                            {
                                this->binWeights[i] = 0._X;
                                this->binDeltaEnergy[i] = 0._X;
                                this->binDeltaWeight[i] = 0._X;
                                this->binLeftBoundary[i] = 0._X;
                            }

                            for(uint16_t i = 0u; i < maxNumNewBins; i++)
                            {
                                this->newBinsLeftBoundary[i] = 0._X;
                                this->newBinsWeights[i] = 0._X;
                            }
                            //} end of debug init
                        }

                        DINLINE static constexpr uint16_t getMaxNumberBins()
                        {
                            return AdaptiveHistogram::maxNumBins;
                        }

                        DINLINE static constexpr uint16_t getMaxNumberNewBins()
                        {
                            return AdaptiveHistogram::maxNumNewBins;
                        }

                        DINLINE uint16_t getNumBins()
                        {
                            return this->numBins;
                        }

                        DINLINE float_X getInitialGridWidth()
                        {
                            return this->initialGridWidth;
                        }

                        DINLINE float_X getLastBinLeftBoundary()
                        {
                            return this->lastBinLeftBoundary;
                        }

                        // return unit: value
                        /** if bin with given index exists, returns it's central energy, otherwise returns 0
                         *
                         * 0 is never a valid central energy since 0 is always a left boundary of a bin
                         *
                         * @param index ... collection index of bin
                         */
                        template<typename T_Acc>
                        DINLINE float_X getEnergyBin(T_Acc& acc, uint16_t index, T_AtomicDataBox atomicDataBox) const
                        {
                            // no need to check for < 0, since uint
                            if(index < this->numBins)
                            {
                                float_X leftBoundary = this->binLeftBoundary[index];
                                float_X binWidth = this->getBinWidth(acc, true, leftBoundary, atomicDataBox);

                                return leftBoundary + binWidth / 2._X;
                            }
                            // index outside range
                            return 0._X;
                        }

                        // returns the left Boundary of the by collection index specified occupied bin
                        DINLINE float_X getLeftBoundaryBin(uint16_t index) const
                        {
                            // no need to check for < 0, since uint
                            if(index < this->numBins)
                            {
                                return this->binLeftBoundary[index];
                            }
                            // index outside range
                            return -1._X;
                        }

                        DINLINE float_X getWeightBin(uint16_t index) const
                        {
                            return this->binWeights[index];
                        }

                        DINLINE float_X getDeltaEnergyBin(uint16_t index) const
                        {
                            return this->binDeltaEnergy[index];
                        }

                        DINLINE float_X getDeltaWeightBin(uint16_t index) const
                        {
                            return this->binDeltaWeight[index];
                        }

                        /** returns collection index of existing bin containing this energy
                         * or maxNumBins otherwise
                         */
                        template<typename T_Acc>
                        DINLINE uint16_t getBinIndex(
                            T_Acc& acc,
                            float_X energy, // unit: argument
                            T_AtomicDataBox atomicDataBox) const
                        {
                            float_X leftBoundary;
                            float_X binWidth;

                            for(uint16_t i = 0; i < this->numBins; i++)
                            {
                                leftBoundary = this->binLeftBoundary[i];
                                binWidth = this->getBinWidth(acc, true, leftBoundary, atomicDataBox);
                                if(this->inBin(true, leftBoundary, binWidth, energy))
                                {
                                    return i;
                                }
                            }
                            return this->maxNumBins;
                        }

                        /** find z, /in Z, iteratively, such that currentBinWidth * 2^z
                         * gives a lower relativeError than the relativeErrorTarget and maximises the
                         * binWidth and return currentBinWidth * 2^z.
                         *
                         * @param directionPositive .. whether the bin faces in positive argument
                         *      direction from the given boundary
                         * @param boundary
                         * @param currentBinWidth ... starting binWidth, the result may be both larger and
                         *      smaller than initial value
                         */
                        template<typename T_Acc>
                        DINLINE float_X getBinWidth(
                            T_Acc& acc,
                            const bool directionPositive, // unitless
                            const float_X boundary, // unit: value
                            T_AtomicDataBox atomicDataBox) const
                        {
                            // debug only, hardcoded binWidths for now, until I have time to rework the error
                            // estimator, Brian Marre, 2021
                            if(boundary < 1)
                                return 0.25_X;
                            else if(boundary < 10)
                                return 1._X;
                            else if(boundary < 50)
                                return 10._X;
                            return boundary / 2;
                        }

                        /** Returns the left boundary of the bin a given argument value x belongs into
                         *
                         * see file description for more information
                         *
                         * @param x ... where object is located that we want to bin
                         * does not change lastBinLeftBoundary
                         */
                        template<typename T_Acc>
                        DINLINE float_X
                        getBinLeftBoundary(T_Acc& acc, float_X const x, T_AtomicDataBox atomicDataBox) const
                        {
                            // wether x is in positive direction with regards to last known
                            // Boundary
                            bool directionPositive = (x >= this->lastLeftBoundary);

                            // always use initial grid Width as starting point iteration
                            float_X currentBinWidth = this->initialGridWidth;

                            // start from prevoius point to reduce seek times
                            //  BEWARE: currently starting point does not change
                            // TODO: seperate this to seeker class with each worker getting it's own
                            //  instance
                            float_X boundary = this->lastLeftBoundary; // unit: argument

                            // preparation for debug access to run time access
                            uint32_t const workerIdx = cupla::threadIdx(acc).x;
                            // debug acess
                            if(workerIdx == 0)
                            {
                                // debug code
                                /*printf(
                                    "        getBinLeftBoundary: directionPositive %s, initialBinWidth: %f, boundary: "
                                    "%f\n",
                                    directionPositive ? "true" : "false",
                                    currentBinWidth,
                                    boundary);*/
                            }

                            // debug only
                            uint16_t loopCounter = 0u;

                            bool inBin = false;
                            while(!inBin)
                            {
                                // debug only
                                loopCounter++;
                                currentBinWidth = getBinWidth(acc, directionPositive, boundary, atomicDataBox);

                                inBin = AdaptiveHistogram::inBin(directionPositive, boundary, currentBinWidth, x);

                                // check wether x is in Bin
                                if(inBin)
                                {
                                    if(directionPositive)
                                        // already left boundary of bin
                                        return boundary;
                                    else
                                        // right boundary of Bin
                                        return boundary - currentBinWidth;
                                }
                                else
                                {
                                    // particle is not in bin
                                    if(directionPositive)
                                    {
                                        // to try the next bin, we shift the boundary one bin to the right
                                        boundary += currentBinWidth;
                                    }
                                    else
                                    {
                                        // ... or left
                                        boundary -= currentBinWidth;
                                    }
                                    // and reset the current width to the initial value to start finding the width with
                                    // the initial value again
                                    currentBinWidth = this->initialGridWidth;
                                }

                                /*printf(
                                    "        getBinLeftBoundary: inBin %s, loopCounter %i, currentBinWidth %f, "
                                    "boundary %f \n",
                                    inBin ? "true" : "false",
                                    loopCounter,
                                    currentBinWidth,
                                    boundary);*/
                            }
                            return boundary;
                        }

                        /// maybe rename to addDelta_Weight and addDelta_Energy to avoid confusion?
                        template<typename T_Acc>
                        DINLINE void addDeltaWeight(T_Acc& acc, uint16_t index, float_X deltaWeight)
                        {
                            cupla::atomicAdd(acc, &(this->binDeltaWeight[index]), deltaWeight);
                        }

                        template<typename T_Acc>
                        DINLINE void addDeltaEnergy(T_Acc& acc, uint16_t index, float_X deltaEnergy)
                        {
                            cupla::atomicAdd(acc, &(this->binDeltaEnergy[index]), deltaEnergy);
                        }

                        // tries to add the specified change to the specified bin, returns true if sucessfull
                        // or false if not enough energy in bin to add delta energy and keep bin weight >=0
                        template<typename T_Acc>
                        DINLINE bool tryRemoveWeightFromBin(T_Acc& acc, uint16_t index, float_X deltaWeight)
                        {
                            if(this->binWeights[index] + this->binDeltaWeight[index] - deltaWeight >= 0._X)
                            {
                                cupla::atomicAdd(acc, &(this->binDeltaWeight[index]), -deltaWeight);
                                return true;
                            }
                            return false;
                        }

                        template<typename T_Acc>
                        DINLINE void removeWeightFromBin(T_Acc& acc, uint16_t index, float_X deltaWeight)
                        {
                            cupla::atomicAdd(acc, &(this->binDeltaWeight[index]), -deltaWeight);
                        }

                        // tries to add the delta energy to the specified bin, returns true if succesfull
                        // or false if not enough energy in bin to add delta energy and keep bin energy >=0
                        template<typename T_Acc>
                        DINLINE bool tryAddEnergyToBin(
                            T_Acc& acc,
                            uint16_t index,
                            float_X deltaEnergy, // unit: argument
                            T_AtomicDataBox atomicDataBox)
                        {
                            if((this->binWeights[index] + this->binDeltaWeight[index])
                                       * this->getEnergyBin(acc, index, atomicDataBox)
                                       * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                                   + deltaEnergy
                               >= 0.0)
                            {
                                cupla::atomicAdd(acc, &(this->binDeltaEnergy[index]), deltaEnergy);
                                return true;
                            }
                            return false;
                            // TODO: think about moving corresponding weight of electrons to lower energy bin
                            //      might be possible if bisn are without gaps, should make histogram filling
                            //      even easier and faster
                        }

                        // add for Argument x, weight to the bin
                        template<typename T_Acc>
                        DINLINE void binObject(
                            T_Acc const& acc,
                            float_X const x,
                            float_X const weight,
                            T_AtomicDataBox atomicDataBox)
                        {
                            // debug only
                            // std::cout << "energy" << x << std::endl;

                            // compute global bin index
                            float_X const binLeftBoundary = this->getBinLeftBoundary(acc, x, atomicDataBox);

                            // debug only
                            // std::cout << "foundLBoundary";

                            // search for bin in collection of existing bins
                            auto const index = findBin(binLeftBoundary);

                            // If the bin was already there, we need to atomically increase
                            // the value, as another thread may contribute to the same bin
                            if(index < maxNumBins) // bin already exists
                            {
                                cupla::atomicAdd(acc, &(this->binWeights[index]), weight);
                            }
                            else
                            {
                                // Otherwise we add it to a collection of new bins
                                // Note: in current dev the namespace is different in cupla
                                // get Index where to deposit it by atomic add to numNewBins
                                // this assures that the same index is not used twice
                                auto newBinIdx
                                    = cupla::atomicAdd<alpaka::hierarchy::Threads>(acc, &numNewBins, uint32_t(1u));
                                if(newBinIdx < maxNumNewBins)
                                {
                                    newBinsWeights[newBinIdx] = weight;
                                    newBinsLeftBoundary[newBinIdx] = binLeftBoundary;
                                }
                                else
                                {
                                    /// If we are here, there were more new bins since the last update
                                    /// call than memory allocated for them
                                    /// Normally, this should not happen
                                    printf(
                                        "ERROR:Too many new bins before call to updateMethod in binObject Method\n");
                                }
                            }
                        }

                        // shift weight to the bin corresponding to energy x
                        // does add the bin to array of new Bins if it does not exist yet.
                        template<typename T_Acc>
                        DINLINE void shiftWeight(
                            T_Acc const& acc,
                            float_X const energy,
                            float_X const weight,
                            T_AtomicDataBox atomicDataBox)
                        {
                            // debug only
                            // std::cout << "                shiftWeightCall" << std::endl;

                            // compute global bin index
                            float_X const binLeftBoundary = this->getBinLeftBoundary(acc, energy, atomicDataBox);

                            // search for bin in collection of existing bins
                            auto const index = findBin(binLeftBoundary);

                            // If the bin was already there, we need to atomically increase
                            // the value, as another thread may contribute to the same bin
                            if(index < maxNumBins) // bin already exists
                            {
                                cupla::atomicAdd(acc, &(this->binDeltaWeight[index]), weight);
                            }
                            else
                            {
                                // Otherwise we add it to a collection of new bins
                                // get Index where to deposit it by atomic add to numNewBins
                                // this assures that the same index is not used twice
                                auto newBinIdx
                                    = cupla::atomicAdd<alpaka::hierarchy::Threads>(acc, &numNewBins, uint32_t(1u));
                                if(newBinIdx < maxNumNewBins)
                                {
                                    newBinsWeights[newBinIdx] = weight;
                                    newBinsLeftBoundary[newBinIdx] = binLeftBoundary;
                                }
                                else
                                {
                                    /// If we are here, there were more new bins since the last update
                                    /// call than memory allocated for them
                                    /// Normally, this should not happen
                                    printf(
                                        "ERROR:Too many new bins before call to updateMethod in shiftWeight Method\n");
                                }
                            }
                        }

                        /** This method moves new bins created by shiftWeight to
                         * the main collection of bins
                         *
                         * Should be called periodically so that we don't run out of memory for new bins
                         * choose maxNumNewBins to fit period and number threads and particle frame size
                         * MUST be called sequentially
                         */
                        DINLINE void updateWithNewShiftBins()
                        {
                            uint32_t const numBinsBeforeUpdate = numBins;
                            for(uint32_t i = 0u; i < this->numNewBins; i++)
                            {
                                // New bins were definitely not present before
                                // But several new bins can be the same actual bin
                                // So we search in the newly added part only
                                auto const index = findBin(this->newBinsLeftBoundary[i], numBinsBeforeUpdate);

                                // If this bin was already added
                                if(index < maxNumBins)
                                    this->binDeltaWeight[index] += this->newBinsWeights[i];
                                else
                                {
                                    if(this->numBins < this->maxNumBins)
                                    {
                                        this->binLeftBoundary[this->numBins] = this->newBinsLeftBoundary[i];
                                        this->binWeights[this->numBins] = 0._X;
                                        this->binDeltaEnergy[this->numBins] = 0._X;
                                        this->binDeltaWeight[this->numBins] = this->newBinsWeights[i];
                                        this->numBins++;
                                    }
                                    else
                                        printf("ERROR: too many bins, max number bins of histogram exceeded\n");
                                    // we ran out of memory, do nothing
                                }
                            }
                            this->numNewBins = 0u;
                        }

                        /** This method moves new bins to the main collection of bins
                         * Should be called periodically so that we don't run out of memory for
                         * new bins
                         * choose maxNumNewBins to fit period and number threads and particle frame
                         * size
                         * MUST be called sequentially
                         */
                        DINLINE void updateWithNewBins()
                        {
                            uint32_t const numBinsBeforeUpdate = numBins;
                            for(uint32_t i = 0u; i < this->numNewBins; i++)
                            {
                                // New bins were definitely not present before
                                // But several new bins can be the same actual bin
                                // So we search in the newly added part only
                                auto const index = findBin(this->newBinsLeftBoundary[i], numBinsBeforeUpdate);

                                // If this bin was already added
                                if(index < maxNumBins)
                                    this->binWeights[index] += this->newBinsWeights[i];
                                else
                                {
                                    if(this->numBins < this->maxNumBins)
                                    {
                                        this->binLeftBoundary[this->numBins] = this->newBinsLeftBoundary[i];
                                        this->binWeights[this->numBins] = this->newBinsWeights[i];
                                        this->binDeltaEnergy[this->numBins] = 0._X;
                                        this->binDeltaWeight[this->numBins] = 0._X;
                                        this->numBins++;
                                    }
                                    else
                                        printf("ERROR: too many bins, max number bins of histogram exceeded\n");
                                    // we ran out of memory, do nothing
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
