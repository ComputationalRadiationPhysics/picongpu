/**
 * Copyright 2013 Felix Schmitt, René Widera, Wolfgang Hoenig
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#ifndef TASKSENDMPI_HPP
#define	TASKSENDMPI_HPP

#include <mpi.h>

#include "memory/buffers/Exchange.hpp"

#include "eventSystem/tasks/MPITask.hpp"

#include "communication/manager_common.h"
#include "communication/ICommunicator.hpp"

namespace PMacc
{

template <class TYPE, unsigned DIM>
class TaskSendMPI : public MPITask
{
public:

    TaskSendMPI(Exchange<TYPE, DIM> *exchange) :
    MPITask(),
    exchange(exchange)
    {

    }

    virtual void init()
    {
        __startTransaction();
        this->request = EnvironmentController::getInstance()
                .getCommunicator().startSend(
                                             exchange->getExchangeType(),
                                             (char*) exchange->getHostBuffer().getPointer(),
                                             exchange->getHostBuffer().getCurrentSize() * sizeof (TYPE),
                                             exchange->getCommunicationTag());
        __endTransaction();
    }

    bool executeIntern()
    {
        if (this->isFinished())
            return true;

        if (this->request == NULL)
            throw std::runtime_error("request was NULL (call executeIntern after freed");

        int flag=0;
        MPI_CHECK(MPI_Test(this->request, &flag, &(this->status)));

        if (flag) //finished
        {
            delete this->request;
            this->request = NULL;
            this->setFinished();
            return true;
        }
        return false;
    }

    virtual ~TaskSendMPI()
    {
        notify(this->myId, SENDFINISHED, NULL);
    }

    void event(id_t, EventType, IEventData*)
    {

    }

    std::string toString()
    {
        return "TaskSendMPI";
    }

private:
    Exchange<TYPE, DIM> *exchange;
    MPI_Request *request;
    MPI_Status status;
};

} //namespace PMacc

#endif	/* TASKSENDMPI_HPP */

