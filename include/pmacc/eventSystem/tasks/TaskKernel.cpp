/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/eventSystem/tasks/TaskKernel.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/Manager.hpp"

namespace pmacc
{
    void TaskKernel::activateChecks()
    {
        canBeChecked = true;
        this->activate();

        Manager::getInstance().addTask(this);
        eventSystem::setTransactionEvent(EventTask(this->getId()));
    }
} // namespace pmacc
