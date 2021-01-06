/* Copyright 2015-2021 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/random/Random.hpp"
#include "pmacc/random/RNGHandle.hpp"
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"
#include "pmacc/dataManagement/ISimulationData.hpp"

namespace pmacc
{
    namespace random
    {
        /**
         * Provider of a per cell random number generator
         *
         * \tparam T_dim Number of dimensions of the grid
         * \tparam T_RNGMethod Method to use for random number generation
         */
        template<uint32_t T_dim, class T_RNGMethod>
        class RNGProvider : public ISimulationData
        {
        public:
            static constexpr uint32_t dim = T_dim;
            typedef T_RNGMethod RNGMethod;
            typedef DataSpace<dim> Space;

        private:
            typedef typename RNGMethod::StateType RNGState;

        public:
            typedef HostDeviceBuffer<RNGState, dim> Buffer;
            typedef typename Buffer::DataBoxType DataBoxType;
            typedef RNGHandle<RNGProvider> Handle;

            template<class T_Distribution>
            struct GetRandomType
            {
                typedef typename T_Distribution::template applyMethod<RNGMethod>::type Distribution;
                typedef Random<Distribution, RNGMethod, Handle> type;
            };

            /**
             * Create the RNGProvider and allocate memory for the given size
             *
             * @param size Size of the grid for which RNGs should be provided
             * @param uniqueId Unique ID for this instance. If none is given the default
             *          (as returned by \ref getName()) is used
             */
            RNGProvider(const Space& size, const std::string& uniqueId = "");
            virtual ~RNGProvider()
            {
                __delete(buffer)
            }
            /**
             * Initializes the random number generators
             * Must be called before usage
             * @param seed Base seed to be used
             */
            void init(uint32_t seed);

            /**
             * Factory method
             * Creates a handle to a state that can be used to create actual RNGs
             *
             * @param id SimulationDataId of the RNGProvider to use. Defaults to the default Id of the type
             */
            static Handle createHandle(const std::string& id = getName());

            /**
             * Factory method
             * Creates functor that creates random numbers with a given distribution
             * Similar to the Handle but can be used directly
             *
             * @param id SimulationDataId of the RNGProvider to use. Defaults to the default Id of the type
             */
            template<class T_Distribution>
            static typename GetRandomType<T_Distribution>::type createRandom(const std::string& id = getName());

            /**
             * Returns the default id for this type
             */
            static std::string getName();
            SimulationDataId getUniqueId() override;
            void synchronize() override;

            /**
             * Return a reference to the buffer containing the states
             * Note: This buffer might be empty
             */
            Buffer& getStateBuffer();

        private:
            /**
             * Gets the device data box
             */
            DataBoxType getDeviceDataBox();

            const Space m_size;
            Buffer* buffer;
            const std::string m_uniqueId;
        };

    } // namespace random
} // namespace pmacc

#include "pmacc/random/RNGProvider.tpp"
