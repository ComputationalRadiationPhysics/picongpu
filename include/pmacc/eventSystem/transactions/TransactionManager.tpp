/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/eventSystem/EventSystem.hpp"

#include <iostream>


namespace pmacc
{
    inline TransactionManager::~TransactionManager() /*noexcept(false)*/
    {
        if(transactions.size() == 0)
            std::cerr << "[PMacc] [TransactionManager] "
                      << "Missing transaction on the stack!" << std::endl;
        else if(transactions.size() > 1)
            std::cerr << "[PMacc] [TransactionManager] "
                      << "Unfinished transactions on the stack" << std::endl;
        transactions.pop();
    }

    inline TransactionManager::TransactionManager()
    {
        startTransaction(EventTask());
    }

    inline TransactionManager::TransactionManager(const TransactionManager&)
    {
    }

    inline void TransactionManager::startTransaction(EventTask serialEvent)
    {
        transactions.push(Transaction(serialEvent));
    }

    inline EventTask TransactionManager::endTransaction()
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling endTransaction on empty transaction stack is not allowed");

        EventTask event = transactions.top().getTransactionEvent();
        transactions.pop();
        return event;
    }

    inline void TransactionManager::startOperation(ITask::TaskType op)
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling startOperation on empty transaction stack is not allowed");

        transactions.top().operation(op);
    }

    inline EventStream* TransactionManager::getEventStream(ITask::TaskType op)
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling startOperation on empty transaction stack is not allowed");

        return transactions.top().getEventStream(op);
    }

    inline EventTask TransactionManager::setTransactionEvent(const EventTask& event)
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling setTransactionEvent on empty transaction stack is not allowed");

        return transactions.top().setTransactionEvent(event);
    }

    inline EventTask TransactionManager::getTransactionEvent()
    {
        if(transactions.size() == 0)
            throw std::runtime_error("Calling getTransactionEvent on empty transaction stack is not allowed");

        return transactions.top().getTransactionEvent();
    }

    inline TransactionManager& TransactionManager::getInstance()
    {
        static TransactionManager instance;
        return instance;
    }


} // namespace pmacc
