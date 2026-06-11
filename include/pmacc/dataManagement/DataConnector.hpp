/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.def"
#include "pmacc/dataManagement/ISimulationData.hpp"
#include "pmacc/debug/PMaccVerbose.hpp"

#include <list>
#include <memory>
#include <stdexcept>

namespace pmacc
{
    namespace detail
    {
        struct Environment;
    } // namespace detail

    /** Singleton class which collects and shares simulation data
     *
     * All members are kept as shared pointers, which allows their factories to
     * be destroyed after sharing ownership with our DataConnector.
     */
    class DataConnector
    {
    private:
        std::list<std::shared_ptr<ISimulationData>>::iterator findId(SimulationDataId id);

    public:
        /** Returns if data with identifier id is shared
         *
         * @param id id of the Dataset to query
         * @return if dataset with id is registered
         */
        bool hasId(SimulationDataId id);

        /** Register a new Dataset and share its ownership.
         *
         * If a Dataset with the same id already exists, a runtime_error is thrown.
         * (Check with DataConnector::hasId when necessary.)
         *
         * @param data simulation data to share ownership
         */
        void share(std::shared_ptr<ISimulationData> const& data);

        /** Register a new Dataset and transfer its ownership.
         *
         * If a Dataset with the same id already exists, a runtime_error is thrown.
         * (Check with DataConnector::hasId when necessary.)
         * The only difference from share() is transfer of ownership.
         *
         * @param data simulation data to transfer ownership
         */
        void consume(std::unique_ptr<ISimulationData> data);

        /** End sharing a dataset with identifier id
         *
         * @param id id of the dataset to remove
         */
        void deregister(SimulationDataId id);

        /** Unshare all associated datasets
         */
        void clean();

        /** Returns shared pointer to managed data.
         *
         * Reference to data in Dataset with identifier id and type TYPE is returned.
         * Increments the reference counter to the dataset specified by id.
         *
         * @tparam TYPE if of the data to load
         * @param id id of the Dataset to load from
         * @return returns a reference to the data of type TYPE
         */
        template<class TYPE>
        std::shared_ptr<TYPE> get(SimulationDataId id)
        {
            auto it = findId(id);

            if(it == datasets.end())
                throw std::runtime_error(getExceptionStringForID("Invalid dataset ID", id));

            log<ggLog::MEMORY>("DataConnector: sharing access to '%1%' (%2% uses)") % id % (it->use_count());
            return std::static_pointer_cast<TYPE>(*it);
        }

    private:
        friend struct detail::Environment;

        static DataConnector& getInstance()
        {
            static DataConnector instance;
            return instance;
        }

        DataConnector() = default;

        virtual ~DataConnector();

        std::string getExceptionStringForID(char const* msg, SimulationDataId id);

        std::list<std::shared_ptr<ISimulationData>> datasets;
    };

} // namespace pmacc
