/**
 * Copyright 2013-2016 Rene Widera, Felix Schmitt
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <map>
#include <sstream>
#include <stdexcept>

#include "dataManagement/Dataset.hpp"
#include "dataManagement/ISimulationData.hpp"
#include "dataManagement/AbstractInitialiser.hpp"
#include "dataManagement/ListSorter.hpp"


namespace PMacc
{

    /**
     * Helper class for DataConnector.
     * Uses std::map<KeyType, ValType> for storing values and has a
     * IDataSorter for iterating over these values according to their keys.
     *
     * \tparam KeyType type of map keys
     * \tparam ValType type of map values
     */
    template<typename KeyType, typename ValType>
    class Mapping
    {
    public:

        /**
         * Destructor.
         *
         * Deletes the IDataSorter and clears the mapping.
         */
        ~Mapping()
        {
            mapping.clear();
        }

        std::map<KeyType, ValType> mapping;
        IDataSorter<KeyType> *sorter;
    };

    /**
     * Singleton class which registers simulation data and tracks their state.
     */
    class DataConnector
    {
    public:

        /**
         * Returns if data with identifier id is registered.
         *
         * @param id id of the Dataset to query
         * @return if dataset with id is registered
         */
        bool hasData(SimulationDataId id)
        {
            return datasets.mapping.find(id) != datasets.mapping.end();
        }

        /**
         * Invalidates all Datasets in the DataConnector.
         */
        void invalidate()
        {
            std::map<SimulationDataId, Dataset*>::iterator iter = datasets.mapping.begin();
            for (; iter != datasets.mapping.end(); ++iter)
                iter->second->invalidate();
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

            if (datasets.sorter->isValid())
            {
                for (SimulationDataId id = datasets.sorter->begin();
                        datasets.sorter->isValid(); id = datasets.sorter->getNext())
                {
                    ISimulationData& data = datasets.mapping[id]->getData();

                    initialiser.init(data, currentStep);

                    if (!datasets.sorter->hasNext())
                        break;
                }
            }

            initialiser.teardown();
        }

        /**
         * Registers a new Dataset with data and identifier id.
         *
         * If a Dataset with identifier id already exists, a runtime_error is thrown.
         * (Check with DataConnector::hasData when necessary.)
         *
         * @param data simulation data to store in the Dataset
         */
        void registerData(ISimulationData &data)
        {
            SimulationDataId id = data.getUniqueId();
            if (hasData(id))
                throw std::runtime_error(getExceptionStringForID("DataConnector dataset ID already exists", id));

            Dataset::DatasetStatus status = Dataset::AUTO_INVALID;

            Dataset * dataset = new Dataset(data, status);
            datasets.mapping[id] = dataset;

            datasets.sorter->add(id);
        }

        /**
         * Returns registered data.
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
        TYPE &getData(SimulationDataId id, bool noSync = false)
        {
            std::map<SimulationDataId, Dataset*>::const_iterator iter = datasets.mapping.find(id);

            if (iter == datasets.mapping.end())
                throw std::runtime_error(getExceptionStringForID("Invalid DataConnector dataset ID", id));

            Dataset * dataset = iter->second;
            if (!noSync)
            {
                dataset->synchronize();
            }

            return (TYPE&) (dataset->getData());
        }

        /**
         * Decrements the reference counter to the data specified by id.
         *
         * @param id id for the dataset previously acquired using getData()
         */
        void releaseData(SimulationDataId)
        {
        }

    private:

        friend class Environment<DIM1>;
        friend class Environment<DIM2>;
        friend class Environment<DIM3>;

        static DataConnector& getInstance()
        {
            static DataConnector instance;
            return instance;
        }

        Mapping<SimulationDataId, Dataset*> datasets;

        DataConnector()
        {
            datasets.sorter = new ListSorter<SimulationDataId > ();
        };

        virtual ~DataConnector()
        {
            std::map<SimulationDataId, Dataset*>::const_iterator iter;
            for (iter = datasets.mapping.begin(); iter != datasets.mapping.end(); iter++)
                delete iter->second;

            if (datasets.sorter != NULL)
            {
                delete datasets.sorter;
                datasets.sorter = NULL;
            }
        }

        std::string getExceptionStringForID(const char *msg, SimulationDataId id)
        {
            std::stringstream stream;
            stream << msg << " (" << id << ")";
            return stream.str();
        }
    };

}

