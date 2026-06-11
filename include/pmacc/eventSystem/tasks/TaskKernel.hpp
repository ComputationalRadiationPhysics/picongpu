/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

namespace pmacc
{
    class TaskKernel : public DeviceTask
    {
    public:
        TaskKernel(std::string kernelName) : DeviceTask(), canBeChecked(false), kernelName(kernelName)
        {
        }

        ~TaskKernel() override
        {
            notify(this->myId, KERNEL, nullptr);
        }

        bool executeIntern() override
        {
            if(canBeChecked)
            {
                return isFinished();
            }
            return false;
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        void activateChecks();

        std::string toString() override
        {
            return std::string("TaskKernel ") + kernelName;
        }

        void init() override
        {
        }

    private:
        bool canBeChecked;
        std::string kernelName;
    };

} // namespace pmacc
