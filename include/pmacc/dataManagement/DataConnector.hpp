/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Axel Huebl, Sergei Bastrakov
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

#include "pmacc/dataManagement/ISimulationData.hpp"
#include "pmacc/dataManagement/AbstractInitialiser.hpp"

#include "pmacc/assert.hpp"

#include <vector>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <utility>


namespace pmacc
{
    /** Singleton class which collects and shares simulation data
     *
     * All members are kept as shared pointers, which allows their factories to
     * be destroyed after sharing ownership with our DataConnector.
     */
    class DataConnector
    {
    private:
        std::list<std::shared_ptr<ISimulationData>>::iterator findId(SimulationDataId id)
        {
            return std::find_if(
                datasets.begin(),
                datasets.end(),
                [&id](std::shared_ptr<ISimulationData> data) -> bool { return data->getUniqueId() == id; });
        }

    public:
        /** Returns if data with identifier id is shared
         *
         * @param id id of the Dataset to query
         * @return if dataset with id is registered
         */
        bool hasId(SimulationDataId id)
        {
            return findId(id) != datasets.end();
        }

        /**
         * Initialises all Datasets using initialiser.
         * After initialising, the Datasets will be invalid.
         *
         * @param initialiser class used for initialising Datasets
         * @param currentStep current simulation step
         */
        void initialise(AbstractInitialiser& initialiser, uint32_t currentStep)
        {
            currentStep = initialiser.setup();

            for(auto& data : datasets)
            {
                initialiser.init(*data, currentStep);
            }

            initialiser.teardown();
        }

        /** Register a new Dataset and share its ownership.
         *
         * If a Dataset with the same id already exists, a runtime_error is thrown.
         * (Check with DataConnector::hasId when necessary.)
         *
         * @param data simulation data to share ownership
         */
        void share(const std::shared_ptr<ISimulationData>& data)
        {
            PMACC_ASSERT(data != nullptr);

            SimulationDataId id = data->getUniqueId();

            log<ggLog::MEMORY>("DataConnector: data shared '%1%'") % id;

            if(hasId(id))
                throw std::runtime_error(getExceptionStringForID("dataset ID already exists", id));

            datasets.push_back(data);
        }

        /** Register a new Dataset and transfer its ownership.
         *
         * If a Dataset with the same id already exists, a runtime_error is thrown.
         * (Check with DataConnector::hasId when necessary.)
         * The only difference from share() is transfer of ownership.
         *
         * @param data simulation data to transfer ownership
         */
        void consume(std::unique_ptr<ISimulationData> data)
        {
            std::shared_ptr<ISimulationData> newOwner(std::move(data));
            share(newOwner);
        }

        /** End sharing a dataset with identifier id
         *
         * @param id id of the dataset to remove
         */
        void deregister(SimulationDataId id)
        {
            const auto it = findId(id);

            if(it == datasets.end())
                throw std::runtime_error(getExceptionStringForID("dataset not found", id));

            log<ggLog::MEMORY>("DataConnector: unshared '%1%' (%2% uses left)") % id % (it->use_count() - 1);

            datasets.erase(it);
        }

        /** Unshare all associated datasets
         */
        void clean()
        {
            log<ggLog::MEMORY>("DataConnector: being cleaned (%1% datasets left to unshare)") % datasets.size();

            // verbose version of: datasets.clear();
            while(!datasets.empty())
            {
                auto it = datasets.rbegin();
                log<ggLog::MEMORY>("DataConnector: unshared '%1%' (%2% uses left)") % (*it)->getUniqueId()
                    % (it->use_count() - 1);
                datasets.pop_back();
            }
        }

        /** Returns shared pointer to managed data.
         *
         * Reference to data in Dataset with identifier id and type TYPE is returned.
         * If the Dataset status in invalid, it is automatically synchronized.
         * Increments the reference counter to the dataset specified by id.
         * This reference has to be released after all read/write operations
         * before the next synchronize()/getData() on this data are done using releaseData().
         *
         * @tparam TYPE if of the data to load
         * @param id id of the Dataset to load from
         * @param noSync indicates that no synchronization should be performed, regardless of dataset status
         * @return returns a reference to the data of type TYPE
         */
        template<class TYPE>
        std::shared_ptr<TYPE> get(
            SimulationDataId id,
            bool noSync = false // @todo invert!
        )
        {
            auto it = findId(id);

            if(it == datasets.end())
                throw std::runtime_error(getExceptionStringForID("Invalid dataset ID", id));

            log<ggLog::MEMORY>("DataConnector: sharing access to '%1%' (%2% uses)") % id % (it->use_count());

            if(!noSync)
            {
                (*it)->synchronize();
            }

            return std::static_pointer_cast<TYPE>(*it);
        }

        /** Indicate a data set gotten temporarily via @see getData is not used anymore
         *
         * @todo not implemented
         *
         * @param id id for the dataset previously acquired using getData()
         */
        void releaseData(SimulationDataId)
        {
        }

    private:
        friend struct detail::Environment;

        static DataConnector& getInstance()
        {
            static DataConnector instance;
            return instance;
        }

        std::list<std::shared_ptr<ISimulationData>> datasets;

        DataConnector(){};

        virtual ~DataConnector()
        {
            log<ggLog::MEMORY>("DataConnector: being destroyed (%1% datasets left to destroy)") % datasets.size();
            clean();
        }

        std::string getExceptionStringForID(const char* msg, SimulationDataId id)
        {
            std::stringstream stream;
            stream << "DataConnector: " << msg << " (" << id << ")";
            return stream.str();
        }
    };

} // namespace pmacc
