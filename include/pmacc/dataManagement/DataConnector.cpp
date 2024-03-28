/* Copyright 2013-2023 Rene Widera, Felix Schmitt, Axel Huebl, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include "pmacc/dataManagement/DataConnector.hpp"

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/Definition.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>


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

    void DataConnector::initialise(AbstractInitialiser& initialiser, uint32_t currentStep)
    {
        currentStep = initialiser.setup();

        for(auto& data : datasets)
        {
            initialiser.init(*data, currentStep);
        }

        initialiser.teardown();
    }


    void DataConnector::share(const std::shared_ptr<ISimulationData>& data)
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
        const auto it = findId(id);

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

    std::string DataConnector::getExceptionStringForID(const char* msg, SimulationDataId id)
    {
        std::stringstream stream;
        stream << "DataConnector: " << msg << " (" << id << ")";
        return stream.str();
    }
} // namespace pmacc
