/*
 * SPDX-FileCopyrightText: Alexander Grund, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dataManagement/ISimulationData.hpp"
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"
#include "pmacc/random/RNGHandle.hpp"
#include "pmacc/random/Random.hpp"
#include "pmacc/types.hpp"

#include <memory>

namespace pmacc
{
    namespace random
    {
        /**
         * Provider of a per cell random number generator
         *
         * @tparam T_dim Number of dimensions of the grid
         * @tparam T_RNGMethod Method to use for random number generation
         */
        template<uint32_t T_dim, class T_RNGMethod>
        class RNGProvider : public ISimulationData
        {
        public:
            static constexpr uint32_t dim = T_dim;
            using RNGMethod = T_RNGMethod;
            using Space = DataSpace<dim>;

        private:
            using RNGState = typename RNGMethod::StateType;

        public:
            using Buffer = HostDeviceBuffer<RNGState, dim>;
            using DataBoxType = typename Buffer::DataBoxType;
            using Handle = RNGHandle<RNGProvider>;

            template<class T_Distribution>
            struct GetRandomType
            {
                using Distribution = typename T_Distribution::template applyMethod<RNGMethod>::type;
                using type = Random<Distribution, RNGMethod, Handle>;
            };

            /**
             * Create the RNGProvider and allocate memory for the given size
             *
             * @param size Size of the grid for which RNGs should be provided
             * @param uniqueId Unique ID for this instance. If none is given the default
             *          (as returned by \ref getName()) is used
             */
            RNGProvider(Space const& size, std::string const& uniqueId = "");

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
            static Handle createHandle(std::string const& id = getName());

            /**
             * Factory method
             * Creates functor that creates random numbers with a given distribution
             * Similar to the Handle but can be used directly
             *
             * @param id SimulationDataId of the RNGProvider to use. Defaults to the default Id of the type
             */
            template<class T_Distribution>
            static typename GetRandomType<T_Distribution>::type createRandom(std::string const& id = getName());

            /**
             * Returns the default id for this type
             */
            static std::string getName();
            SimulationDataId getUniqueId() override;
            void synchronize() override;

            //! Synchronize device data with host data
            void syncToDevice();

            /**
             * Return a reference to the buffer containing the states
             * Note: This buffer might be empty
             */
            Buffer& getStateBuffer();

            //! Get size of the internal buffer
            HINLINE Space getSize() const;

        private:
            /**
             * Gets the device data box
             */
            DataBoxType getDeviceDataBox();

            Space const m_size;
            std::unique_ptr<Buffer> buffer;
            std::string const m_uniqueId;
        };

    } // namespace random
} // namespace pmacc

#include "pmacc/random/RNGProvider.tpp"
