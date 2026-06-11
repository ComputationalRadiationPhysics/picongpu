/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/eventSystem/eventSystem.hpp"

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

    EventTask setTransactionEvent(EventTask const& event)
    {
        return TransactionManager::getInstance().setTransactionEvent(event);
    }

    EventTask getTransactionEvent()
    {
        return TransactionManager::getInstance().getTransactionEvent();
    }

    Queue* getComputeDeviceQueue(ITask::TaskType op)
    {
        return TransactionManager::getInstance().getComputeDeviceQueue(op);
    }
} // namespace pmacc::eventSystem
