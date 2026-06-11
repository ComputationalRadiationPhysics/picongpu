/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/eventSystem/waitForAllTasks.hpp"

#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/transactions/TransactionManager.hpp"

namespace pmacc::eventSystem
{
    void waitForAllTasks()
    {
        Manager::getInstance().waitForAllTasks();
    }
} // namespace pmacc::eventSystem
