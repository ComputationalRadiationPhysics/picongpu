/* 
 * File:   TaskKernel.tpp
 * Author: schuma
 *
 * Created on 30. Januar 2014, 13:30
 */

#include "TaskKernel.hpp"

#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/EventSystem.hpp"

namespace PMacc {

    TaskKernel::TaskKernel(std::string kernelName) :
    StreamTask(),
    kernelName(kernelName),
    canBeChecked(false) {
    }

    TaskKernel::~TaskKernel() {
        notify(this->myId, KERNEL, NULL);
    }

    bool TaskKernel::executeIntern() throw (std::runtime_error) {
        if (canBeChecked) {
            return isFinished();
        }
        return false;
    }

    void TaskKernel::event(id_t, EventType, IEventData*) {
    }

    void TaskKernel::activateChecks() {
        canBeChecked = true;
        this->activate();

        Environment<>::getInstance().getManager().addTask(this);
        __setTransactionEvent(EventTask(this->getId()));
    }

    std::string TaskKernel::toString() {
        return std::string("TaskKernel ") + kernelName;
    }

    void TaskKernel::init() {
    }
}

