/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/dataManagement/DataConnector.hpp"

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/Definition.hpp"

#include <algorithm>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pmacc
{
    std::list<std::shared_ptr<ISimulationData>>::iterator DataConnector::findId(SimulationDataId id)
    {
        return std::find_if(
            datasets.begin(),
            datasets.end(),
            [&id](std::shared_ptr<ISimulationData> data) -> bool { return data->getUniqueId() == id; });
    }

    bool DataConnector::hasId(SimulationDataId id)
    {
        return findId(id) != datasets.end();
    }

    void DataConnector::share(std::shared_ptr<ISimulationData> const& data)
    {
        PMACC_ASSERT(data != nullptr);

        SimulationDataId id = data->getUniqueId();

        log<ggLog::MEMORY>("DataConnector: data shared '%1%'") % id;

        if(hasId(id))
            throw std::runtime_error(getExceptionStringForID("dataset ID already exists", id));

        datasets.push_back(data);
    }

    void DataConnector::consume(std::unique_ptr<ISimulationData> data)
    {
        std::shared_ptr<ISimulationData> newOwner(std::move(data));
        share(newOwner);
    }

    void DataConnector::deregister(SimulationDataId id)
    {
        auto const it = findId(id);

        if(it == datasets.end())
            throw std::runtime_error(getExceptionStringForID("dataset not found", id));

        log<ggLog::MEMORY>("DataConnector: unshared '%1%' (%2% uses left)") % id % (it->use_count() - 1);

        datasets.erase(it);
    }

    void DataConnector::clean()
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

    DataConnector::~DataConnector()
    {
        log<ggLog::MEMORY>("DataConnector: being destroyed (%1% datasets left to destroy)") % datasets.size();
        clean();
    }

    std::string DataConnector::getExceptionStringForID(char const* msg, SimulationDataId id)
    {
        std::stringstream stream;
        stream << "DataConnector: " << msg << " (" << id << ")";
        return stream.str();
    }
} // namespace pmacc
