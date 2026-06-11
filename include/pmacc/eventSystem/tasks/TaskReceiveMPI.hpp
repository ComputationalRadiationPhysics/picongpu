/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"

#include <memory>

#include <mpi.h>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskReceiveMPI : public MPITask
    {
    public:
        TaskReceiveMPI(Exchange<TYPE, DIM>* exchange) : MPITask(), exchange(exchange)
        {
        }

        void init() override
        {
            auto cPtr = exchange->getCPtrCapacity();

            this->request = Environment<DIM>::get().EnvironmentController().getCommunicator().startReceive(
                exchange->getExchangeType(),
                cPtr.asCharPtr(),
                cPtr.sizeInBytes(),
                exchange->getCommunicationTag());
        }

        bool executeIntern() override
        {
            if(this->isFinished())
                return true;

            if(this->request == nullptr)
                throw std::runtime_error("request was nullptr (call executeIntern after freed");

            int flag = 0;
            MPI_CHECK(MPI_Test(this->request, &flag, &(this->status)));

            if(flag) // finished
            {
                delete this->request;
                this->request = nullptr;
                setFinished();
                return true;
            }
            return false;
        }

        ~TaskReceiveMPI() override
        {
            //! \todo this make problems because we send bytes and not combined types
            int recv_data_count;
            MPI_CHECK_NO_EXCEPT(MPI_Get_count(&(this->status), MPI_CHAR, &recv_data_count));


            std::unique_ptr<IEventData> edata = std::make_unique<EventDataReceive>(nullptr, recv_data_count);

            notify(this->myId, RECVFINISHED, edata.get()); /*add notify her*/
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return std::string("TaskReceiveMPI exchange type=") + std::to_string(exchange->getExchangeType());
        }

    private:
        Exchange<TYPE, DIM>* exchange;
        MPI_Request* request;
        MPI_Status status;
    };

} // namespace pmacc
