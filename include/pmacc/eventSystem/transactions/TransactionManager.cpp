/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

    EventTask TransactionManager::setTransactionEvent(EventTask const& event)
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
