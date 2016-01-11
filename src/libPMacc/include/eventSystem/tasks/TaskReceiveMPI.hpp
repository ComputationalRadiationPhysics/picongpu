/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#pragma once

#include "communication/manager_common.h"
#include "communication/ICommunicator.hpp"
#include "eventSystem/tasks/MPITask.hpp"
#include "memory/buffers/Exchange.hpp"

#include <mpi.h>

namespace PMacc
{

template <class TYPE, unsigned DIM>
class TaskReceiveMPI : public MPITask
{
public:

    TaskReceiveMPI(Exchange<TYPE, DIM> *exchange) :
    MPITask(),
    exchange(exchange)
    {

    }

    virtual void init()
    {
        this->request = Environment<DIM>::get().EnvironmentController()
                .getCommunicator().startReceive(
                                                exchange->getExchangeType(),
                                                (char*) exchange->getHostBuffer().getBasePointer(),
                                                exchange->getHostBuffer().getDataSpace().productOfComponents() * sizeof (TYPE),
                                                exchange->getCommunicationTag());
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
            setFinished();
            return true;
        }
        return false;
    }

    virtual ~TaskReceiveMPI()
    {
        //\\todo: this make problems because we send bytes and not combined types
        int recv_data_count;
        MPI_CHECK(MPI_Get_count(&(this->status), MPI_CHAR, &recv_data_count));


        IEventData *edata = new EventDataReceive(NULL, recv_data_count);

        notify(this->myId, RECVFINISHED, edata); /*add notify her*/
        __delete(edata);

    }

    void event(id_t, EventType, IEventData*)
    {


    }

    std::string toString()
    {
        return "TaskReceiveMPI";
    }

private:
    Exchange<TYPE, DIM> *exchange;
    MPI_Request *request;
    MPI_Status status;
};

} //namespace PMacc

