/* Copyright 2023 Rene Widera
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


#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/transactions/TransactionManager.hpp"

namespace pmacc::eventSystem
{
    void startTransaction(EventTask serialEvent)
    {
        TransactionManager::getInstance().startTransaction(serialEvent);
    }

    EventTask endTransaction()
    {
        return TransactionManager::getInstance().endTransaction();
    }

    void startOperation(ITask::TaskType op)
    {
        TransactionManager::getInstance().startOperation(op);
    };

    EventTask setTransactionEvent(const EventTask& event)
    {
        return TransactionManager::getInstance().setTransactionEvent(event);
    }

    EventTask getTransactionEvent()
    {
        return TransactionManager::getInstance().getTransactionEvent();
    }

    EventStream* getEventStream(ITask::TaskType op)
    {
        return TransactionManager::getInstance().getEventStream(op);
    }

    void waitForAllTasks()
    {
        Manager::getInstance().waitForAllTasks();
    }
} // namespace pmacc::eventSystem
