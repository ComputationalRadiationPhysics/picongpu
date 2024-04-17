/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz
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


#include "pmacc/eventSystem/transactions/TransactionManager.hpp"

#include "pmacc/Environment.def"

#include <iostream>


namespace pmacc
{
    TransactionManager::~TransactionManager() /*noexcept(false)*/
    {
        if(transactions.size() == 0)
            std::cerr << "[PMacc] [TransactionManager] "
                      << "Missing transaction on the stack!" << std::endl;
        else if(transactions.size() > 1)
            std::cerr << "[PMacc] [TransactionManager] "
                      << "Unfinished transactions on the stack" << std::endl;
        transactions.pop();
    }

    TransactionManager::TransactionManager()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isDeviceSelected(),
            "No device selected you must call Environment< DIM >::initDevices(...) before this method.");

        startTransaction(EventTask());
    }

    void TransactionManager::startTransaction(EventTask serialEvent)
    {
        transactions.push(Transaction(serialEvent));
    }

    EventTask TransactionManager::endTransaction()
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling endTransaction on empty transaction stack is not allowed");

        EventTask event = transactions.top().getTransactionEvent();
        transactions.pop();
        return event;
    }

    void TransactionManager::startOperation(ITask::TaskType op)
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling startOperation on empty transaction stack is not allowed");

        transactions.top().operation(op);
    }

    Queue* TransactionManager::getComputeDeviceQueue(ITask::TaskType op)
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling startOperation on empty transaction stack is not allowed");

        return transactions.top().getComputeDeviceQueue(op);
    }

    EventTask TransactionManager::setTransactionEvent(const EventTask& event)
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling setTransactionEvent on empty transaction stack is not allowed");

        return transactions.top().setTransactionEvent(event);
    }

    EventTask TransactionManager::getTransactionEvent()
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling getTransactionEvent on empty transaction stack is not allowed");

        return transactions.top().getTransactionEvent();
    }

} // namespace pmacc
