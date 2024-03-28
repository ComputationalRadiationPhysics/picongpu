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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file This file implements an adaptive Histogram, starting from argument 0
 *
 * Basic idea: 1-dimensional infinite histogram, with sparsely populated bins.
 * BinWidths are not uniform, but indirectly defined by a target value of a
 * binning width dependent relative error, therefore an error approximator
 * which is independent of the actual bin content.
 *
 * Only populated bins are stored in memory.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <alpaka/alpaka.hpp>

#include <utility>

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
                    /** @class Adaptive Histogram implementation, see file for principles
                     *
                     * TemplateParameters:
                     * -------------------
                     *  @tparam T_AtomicDataBox type of data container for atomic input data
                     *  @tparam T_maxNumberBins maximum number of bins of the histogram
                     *  @tparam T_maxNumberNewBins maximum number of new bins before
                     *       updateWithNewBins must be called by ONE thread
                     *       BEWARE: causes invalid memory access if set incorrectly
                     */

                    template<uint32_t T_maxNumberBins, uint32_t T_maxNumberNewBins, typename T_AtomicDataBox>
                    struct AdaptiveHistogram
                    {
                    private:
                        //{ members
                        //! renaming of T_maxNumberBins
                        constexpr static uint32_t maxNumBins = T_maxNumberBins;
                        //! renaming of T_maxNumberNewBins
                        constexpr static uint32_t maxNumNewBins = T_maxNumberNewBins;

                        /** histogram storage, weights of particles in bin
                         *
                         * Is updated by the fillHistogram Functor once per PIC step.
                         */
                        float_X binWeight[maxNumBins];
                        /** histogram storage, accumulated change in particle Energy
                         *
                         * for the current time step, is updated by the atomicPhysics solver
                         */
                        float_X binDeltaEnergy[maxNumBins];
                        /** histogram storage, accumulated weight added to this bin, due to interactions
                         *
                         * for the current time step, is updated by the atomicPhysics solver
                         */
                        float_X binDeltaWeight[maxNumBins];
                        //}
                        /** histogram storage, left boundary of bin
                         *
                         * This together with the implicitly defined bin width, calculated
                         * at runtime via call of getBinWidth(), completely defines the bin
                         */
                        float_X binLeftBoundary[maxNumBins];


                        /** number of bins occupied currently occupied
                         *
                         * NOTE: must be <= T_maxNumberBins, all further bins are dropped!
                         */
                        uint16_t numBins;

                        //{new bins storage, since last call to update method
                        /** storage for new Bins since last call to updateWithNewBins, left boundary of bin
                         *
                         * This together with the implicitly defined bin width, calculated
                         * at runtime via call of getBinWidth(), completely defines the bin
                         */
                        float_X newBinsLeftBoundary[maxNumNewBins];
                        //! storage for new Bins since last call to updateWithNewBins, weight in bin
                        float_X newBinsWeights[maxNumNewBins];
                        /* No <newBinsDeltaEnergy> necessary, since all electrons are
                         * already binned for this atomic physics solver step
                         */

                        /** number of new bins, since last call to update method,
                         *
                         * NOTE: must be <= T_maxNumberNewBins, all further bins are dropped!
                         */
                        uint32_t numNewBins;
                        //}

                        //{ administration data
                        /** last left boundary of a bin used
                         *
                         * reference point of histogram, always boundary of a bin in the
                         * histogram
                         *
                         * NOTE: in current implementation not used
                         *
                         * Unit: Argument
                         */
                        float_X lastLeftBoundary; // Unit: Argument

                        /** relative error target used to choose bin size
                         *
                         * target value of relative Error of a histogram bin,
                         * bin width choosen such that relativeError is close to target
                         *
                         * Unit: varies depending on quantity stored in histogram and error
                         *  estimator
                         */
                        float_X relativeErrorTarget; // Unit: varies

                    public:
                        //! initial binWidth used, defines initial global grid, Unit: argument
                        float_X initialGridWidth; // Unit: Argument
                        //}

                    private:
                        /// @todo : replace linear search, by ordering bins
                        /** tries to find bin with given left boundary
                         *
                         * Tries to find binLeftBoundary in the collection,
                         * starting at the given collection index and return the collection
                         * index.
                         *
                         * @return Returns index in the binIndices array when present or
                         *      maxNumBin when not present, maxNumBin is never a valid
                         *      index of the collection
                         *
                         * @param leftBoundary left boundary of bin to search, Unit: argument
                         * @param startIndex index from which to start the search,
                         *  only searches in collection entries with an index higher than
                         *  the given start index
                         */
                        DINLINE uint16_t findBin(
                            float_X leftBoundary, // Unit: Argument
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

                        //! checks whether Bin exists
                        DINLINE bool hasBin(float_X leftBoundary) const
                        {
                            auto const index = findBin(leftBoundary);
                            return (index < maxNumBins);
                        }

                        /** helper function, return center of bin
                         *
                         * case directionPositive == true:  [boundary, boundary + binWidth)
                         * case directionPositive == false: [boundary - binWidth,boundary)
                         *
                         * @param directionPositive direction positive indicates whether
                         *   the left boundary(true) or the right boundary(false) is given
                         * @param boundary left or right boundary of bin, Unit: Argument
                         * @param binWidth width of bin, Unit: Argument
                         */
                        DINLINE static float_X centerBin(
                            bool const directionPositive,
                            float_X const boundary,
                            float_X const binWidth)
                        {
                            if(directionPositive)
                                return boundary + binWidth / 2._X;
                            else
                                return boundary - binWidth / 2._X;
                        }

                        /** helper function, is x in Bin?
                         *
                         *  case directionPositive == true:    [boundary, boundary + binWidth)
                         *  case directionPositive == false:   [boundary - binWidth,boundary)
                         *
                         * @param directionPositive direction positive indicates whether
                         *   the left boundary(true) or the right boundary(false) is given
                         * @param boundary left or right boundary of bin, Unit: Argument
                         * @param binWidth width of bin, Unit: Argument
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
                        /** Has to be called by ONE thread once before use
                         * of this object to correctly initialize the histogram.
                         *
                         * BEWARE: Assumptions for input parameters must be full filled!
                         *
                         * @param relativeErrorTarget ... should be >0,
                         *      maximum relative Error of Histogram Bin
                         * @param initialGridWidth ... should be >0,
                         *      starting bin width of algorithm
                         * @param relativeError ... relative should be monoton rising with rising binWidth
                         */
                        DINLINE void init(float_X relativeErrorTarget, float_X initialGridWidth)
                        {
                            // init histogram empty
                            this->numBins = 0u;
                            this->numNewBins = 0u;

                            // init with 0 as reference point
                            this->lastLeftBoundary = 0._X;

                            // init adaptive bin width algorithm parameters
                            this->relativeErrorTarget = relativeErrorTarget;
                            this->initialGridWidth = initialGridWidth;

                            /// @todo : make this debug mode only
                            // since we are filling memory we are never touching (if everything works)

                            //{ start of debug init
                            for(uint16_t i = 0u; i < maxNumBins; i++)
                            {
                                this->binWeight[i] = 0._X;
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

                        //! get value of template parameter T_maxNumberBins
                        DINLINE static constexpr uint16_t getMaxNumberBins()
                        {
                            return AdaptiveHistogram::maxNumBins;
                        }

                        //! get value of template parameter T_maxNumberNewBins
                        DINLINE static constexpr uint16_t getMaxNumberNewBins()
                        {
                            return AdaptiveHistogram::maxNumNewBins;
                        }

                        //! get current number of occupied Bins
                        DINLINE uint16_t getNumBins()
                        {
                            return this->numBins;
                        }

                        //! get initial grid width, Unit: Argument, may change in the future
                        DINLINE float_X getInitialGridWidth()
                        {
                            return this->initialGridWidth;
                        }

                        //! get last Bin Left Boundary calculated, currently unused, Unit: Argument
                        DINLINE float_X getLastBinLeftBoundary()
                        {
                            return this->lastBinLeftBoundary;
                        }

                        /** if bin with given index exists, returns it's central energy, otherwise returns 0
                         *
                         * 0 is never a valid central energy since 0 is always a left boundary of a bin
                         *
                         * @param index collection index of bin
                         * @param acc accelerator config for on accelerator execution
                         * @param atomicDataBox atomic data for relative error estimation
                         *
                         * @return return Unit: Value
                         */
                        template<typename T_Worker>
                        DINLINE float_X
                        getEnergyBin(T_Worker const& worker, uint16_t index, T_AtomicDataBox atomicDataBox) const
                        {
                            // no need to check for < 0, since uint
                            if(index < this->numBins)
                            {
                                float_X leftBoundary = this->binLeftBoundary[index];
                                float_X binWidth = this->getBinWidth(worker, true, leftBoundary, atomicDataBox);

                                return leftBoundary + binWidth / 2._X;
                            }
                            // index outside range
                            return 0._X;
                        }

                        /** return the left boundary of the specified bin
                         *
                         * Bin is specified by collection index.
                         *
                         * @return If an invalid index is given returns -1.0
                         */
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

                        /** returns weight of Bin with given collection index
                         *
                         * BEWARE: does not check if the bin is occupied, or exists at all.
                         *  Will cause invalid memory access if misused.
                         *
                         * Leave alone unless you know what you are doing!
                         */
                        DINLINE float_X getWeightBin(uint16_t index) const
                        {
                            return this->binWeight[index];
                        }

                        /** returns current deltaEnergy of Bin with given collection index
                         *
                         * BEWARE: does not check if the bin is occupied, or exists at all.
                         *  Will cause invalid memory access if misused.
                         *
                         * Leave alone unless you know what you are doing!
                         */
                        DINLINE float_X getDeltaEnergyBin(uint16_t index) const
                        {
                            return this->binDeltaEnergy[index];
                        }

                        /** returns current deltaWeight of Bin with given collection index
                         *
                         * BEWARE: does not check if the bin is occupied, or exists at all.
                         *  Will cause invalid memory access if misused.
                         *
                         * Leave alone unless you know what you are doing!
                         */
                        DINLINE float_X getDeltaWeightBin(uint16_t index) const
                        {
                            return this->binDeltaWeight[index];
                        }

                        /** get collection index, of bin containing given energy
                         *
                         * @param energy given energy, Unit: Argument
                         * @param acc accelerator config for on accelerator execution
                         * @param atomicDataBox atomic data for relative error estimation
                         *
                         * @return returns collection index of existing bin containing
                         * this energy or maxNumBins otherwise
                         */
                        template<typename T_Worker>
                        DINLINE uint16_t getBinIndex(
                            T_Worker const& worker,
                            float_X energy, // Unit: Argument
                            T_AtomicDataBox atomicDataBox) const
                        {
                            float_X leftBoundary;
                            float_X binWidth;

                            for(uint16_t i = 0; i < this->numBins; i++)
                            {
                                leftBoundary = this->binLeftBoundary[i];
                                binWidth = this->getBinWidth(worker, true, leftBoundary, atomicDataBox);
                                if(this->inBin(true, leftBoundary, binWidth, energy))
                                {
                                    return i;
                                }
                            }
                            return this->maxNumBins;
                        }

                        /** calculate the bin width of the a bin starting from the specified boundary
                         *
                         * Currently uses hard coded values for the beginning and
                         * exponentially growing values after running out.
                         *
                         * In Future will be done as follows:
                         *
                         * The correct bin width is prescribed by the given error estimator,
                         * the relativeErrorTarget, and the initial gridWidth.
                         *
                         * We are essentially searching for a z, \in \Z, iteratively, such
                         * that currentBinWidth * 2^z as input to the given error approximator
                         * results in a relative error lower than the relativeErrorTarget,
                         * while maximizing the bin width.
                         *
                         * The currentBinWidth * 2^z is then returned as the correct bin width.
                         *
                         * This results in a well defined set of bin boundaries, if a common
                         * reference point is used and bins are consecutive.
                         *
                         * @param directionPositive whether the bin faces in positive argument
                         *      direction from the given boundary, see inBin() and centerBin()
                         *      for more information.
                         * @param boundary boundary of bin, Unit: Value
                         * @param currentBinWidth starting binWidth, the result may be both
                         *      larger and smaller than initial value, Unit: Argument
                         */
                        template<typename T_Worker>
                        DINLINE float_X getBinWidth(
                            T_Worker const& worker,
                            const bool directionPositive, // unitless
                            const float_X boundary, // Unit: value
                            T_AtomicDataBox atomicDataBox) const
                        {
                            //! debug only, @todo hardcoded binWidths for now, until I have time to rework the error
                            //! estimator, Brian Marre, 2021
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
                         * see file description and getBinWidth() for more information
                         *
                         * NOTE: current implementation does not change lastBinLeftBoundary
                         *  but future implementations will use this to save time.
                         *
                         * @param x where object is located that we want to bin
                         * @param acc accelerator config for on accelerator execution
                         * @param atomicDataBox atomic data for relative error estimation
                         */
                        template<typename T_Worker>
                        DINLINE float_X getBinLeftBoundary(
                            T_Worker const& worker,
                            float_X const x,
                            T_AtomicDataBox atomicDataBox) const
                        {
                            // whether x is in positive direction with regards to last known
                            // Boundary
                            bool directionPositive = (x >= this->lastLeftBoundary);

                            // always use initial grid Width as starting point iteration
                            float_X currentBinWidth = this->initialGridWidth;

                            // start from prevoius point to reduce seek times
                            //  BEWARE: currently starting point does not change
                            //! @todo : separate this to seeker class with each worker
                            //! getting it's own instance, probably context variable
                            float_X boundary = this->lastLeftBoundary; // Unit: Argument

                            bool inBin = false;
                            while(!inBin)
                            {
                                currentBinWidth = getBinWidth(worker, directionPositive, boundary, atomicDataBox);

                                inBin = AdaptiveHistogram::inBin(directionPositive, boundary, currentBinWidth, x);

                                // check whether x is in Bin
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
                            }
                            return boundary;
                        }

                        /// @todo : maybe rename to addDelta_Weight and addDelta_Energy to avoid confusion?
                        /** add/subtract weight to/from the change accumulation of weights of a bin
                         *
                         * Used to update the histogram by the rate solver upon doing an
                         * atomic physics transitions, shifting weight from one bin to
                         * another already existing bin.
                         *
                         * BEWARE: Does not check if enough weight present.
                         *  See tryRemoveWeightFromBin() for a version which does check.
                         * NOTE: only differs in sign convention from removeWeightFromBin()
                         *
                         * @param index collection index of the bin to which to add the weight
                         * @param deltaWeight weight to add, can be negative
                         * @param acc accelerator config for on accelerator execution
                         */
                        template<typename T_Worker>
                        DINLINE void addDeltaWeight(T_Worker const& worker, uint16_t index, float_X deltaWeight)
                        {
                            alpaka::atomicAdd(worker.getAcc(), &(this->binDeltaWeight[index]), deltaWeight);
                        }

                        /** record energy taken/added from/to a given bin
                         *
                         * Used to record how much energy was taken from a bin through
                         * atomic transitions by the atomic physics solver.
                         *
                         * BEWARE: Does not check if enough energy present.
                         *  See tryRemoveEnergyFromBin() for a version which does check.
                         *
                         * @param index collection index of the bin to which to add the weight
                         * @param deltaWeight weight to add, can be negative
                         * @param worker lockstep worker
                         */
                        template<typename T_Worker>
                        DINLINE void addDeltaEnergy(T_Worker const& worker, uint16_t index, float_X deltaEnergy)
                        {
                            alpaka::atomicAdd(worker.getAcc(), &(this->binDeltaEnergy[index]), deltaEnergy);
                        }

                        /** tries to reduce the weight of the specified bin by deltaWeight,
                         * unless this would result in a negative weight.
                         *
                         * @param index collection index of the bin to which to add the weight
                         * @param deltaWeight weight to remove
                         * @param worker lockstep worker
                         *
                         * @return returns true if successful, or false if not enough weight
                         *  in bin to add delta weight and keep bin weight >=0
                         */
                        template<typename T_Worker>
                        DINLINE bool tryRemoveWeightFromBin(
                            T_Worker const& worker,
                            uint16_t index,
                            float_X deltaWeight)
                        {
                            if(this->binWeight[index] + this->binDeltaWeight[index] - deltaWeight >= 0._X)
                            {
                                // updates deltaWeight instead of binWeight
                                alpaka::atomicAdd(worker.getAcc(), &(this->binDeltaWeight[index]), -deltaWeight);
                                return true;
                            }
                            return false;
                        }

                        //! same as addDeltaWeight() but different sign convention
                        template<typename T_Worker>
                        DINLINE void removeWeightFromBin(T_Worker const& worker, uint16_t index, float_X deltaWeight)
                        {
                            alpaka::atomicAdd(worker.getAcc(), &(this->binDeltaWeight[index]), -deltaWeight);
                        }

                        /** tries to add the delta energy to the specified bin, unless this
                         * would result in a negative energy.
                         *
                         * @param index collection index of the bin to which to add the weight
                         * @param deltaEnergy energy to add
                         * @param worker lockstep worker
                         *
                         * @return returns true if successful, or false if not enough energy
                         * in bin to add delta energy and keep bin energy >=0
                         */
                        template<typename T_Worker>
                        DINLINE bool tryAddEnergyToBin(
                            T_Worker const& worker,
                            uint16_t index,
                            float_X deltaEnergy, // Unit: Argument
                            T_AtomicDataBox atomicDataBox)
                        {
                            if((this->binWeight[index] + this->binDeltaWeight[index])
                                       * this->getEnergyBin(worker, index, atomicDataBox)
                                       * static_cast<float_X>(
                                           picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE)
                                   + deltaEnergy
                               >= 0.0)
                            {
                                alpaka::atomicAdd(worker.getAcc(), &(this->binDeltaEnergy[index]), deltaEnergy);
                                return true;
                            }
                            return false;
                            /// @todo : think about moving corresponding weight of electrons to lower energy bin
                            ///      might be possible if bins are without gaps, should make histogram filling
                            ///      even easier and faster
                        }

                        /** add for Argument x, weight to the bin containing the argument x
                         *
                         * @param x where object is located that we want to bin
                         * @param weight weight of the object to bin
                         * @param worker lockstep worker
                         * @param atomicDataBox atomic data for relative error estimation
                         */
                        template<typename T_Worker>
                        DINLINE void binObject(
                            T_Worker const& worker,
                            float_X const x,
                            float_X const weight,
                            T_AtomicDataBox atomicDataBox)
                        {
                            // compute global bin index
                            float_X const binLeftBoundary = this->getBinLeftBoundary(worker, x, atomicDataBox);

                            // search for bin in collection of existing bins
                            auto const index = findBin(binLeftBoundary);

                            // If the bin was already there, we need to atomically increase
                            // the value, as another thread may contribute to the same bin
                            if(index < maxNumBins) // bin already exists
                            {
                                alpaka::atomicAdd(worker.getAcc(), &(this->binWeight[index]), weight);
                            }
                            else
                            {
                                // Otherwise we add it to a collection of new bins

                                // get Index where to deposit it by atomic add to numNewBins
                                // this assures that the same index is not used twice
                                auto newBinIdx = alpaka::atomicAdd(
                                    worker.getAcc(),
                                    &numNewBins,
                                    uint32_t(1u),
                                    alpaka::hierarchy::Threads{});
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
                                    PMACC_ASSERT_MSG(
                                        false,
                                        "ERROR:Too many new bins before call to updateMethod in binObject Method");
                                    printf(
                                        "ERROR:Too many new bins before call to updateMethod in binObject Method\n");
                                }
                            }
                        }


                        /** shift weight to the bin corresponding to energy x
                         *
                         * Does add the bin to array of new Bins if it does not exist yet.
                         *
                         * @param energy energy where the weight is shifted to, Unit: argument
                         * @param weight weight to shift
                         * @param worker lockstep worker
                         * @param atomicDataBox atomic data for relative error estimation
                         */
                        template<typename T_Worker>
                        DINLINE void shiftWeight(
                            T_Worker const& worker,
                            float_X const energy,
                            float_X const weight,
                            T_AtomicDataBox atomicDataBox)
                        {
                            // compute bin left boundary for energy
                            float_X const binLeftBoundary = this->getBinLeftBoundary(worker, energy, atomicDataBox);

                            // search for bin in collection of existing bins
                            auto const index = findBin(binLeftBoundary);

                            // If the bin was already there, we need to atomically increase
                            // the value, as another thread may contribute to the same bin
                            if(index < maxNumBins) // bin already exists
                            {
                                alpaka::atomicAdd(worker.getAcc(), &(this->binDeltaWeight[index]), weight);
                            }
                            else
                            {
                                // Otherwise we add it to a collection of new bins
                                // get Index where to deposit it by atomic add to numNewBins
                                // this assures that the same index is not used twice
                                auto newBinIdx = alpaka::atomicAdd(
                                    worker.getAcc(),
                                    &numNewBins,
                                    uint32_t(1u),
                                    alpaka::hierarchy::Threads{});
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
                                    PMACC_ASSERT_MSG(
                                        false,
                                        "ERROR:Too many new bins before call to updateMethod in binObject Method");
                                    printf(
                                        "ERROR:Too many new bins before call to updateMethod in shiftWeight Method\n");
                                }
                            }
                        }

                        /** Move new bins created by shiftWeight to the main collection of bins
                         *
                         * Should be called periodically so that we don't run out of memory for new bins
                         * choose maxNumNewBins to fit period and number threads and particle frame size
                         * MUST be called sequentially.
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
                                        this->binWeight[this->numBins] = 0._X;
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

                        /** Move new bins to the main collection of bins
                         *
                         * Should be called periodically so that we don't run out of memory for
                         * new bins, choose maxNumNewBins to fit period and number threads
                         * and particle frame size.
                         * MUST be called sequentially.
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
                                    this->binWeight[index] += this->newBinsWeights[i];
                                else
                                {
                                    if(this->numBins < this->maxNumBins)
                                    {
                                        this->binLeftBoundary[this->numBins] = this->newBinsLeftBoundary[i];
                                        this->binWeight[this->numBins] = this->newBinsWeights[i];
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
