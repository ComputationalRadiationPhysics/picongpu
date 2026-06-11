/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"

#include <mpi.h>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskSendMPI : public MPITask
    {
    public:
        TaskSendMPI(Exchange<TYPE, DIM>* exchange) : MPITask(), exchange(exchange)
        {
        }

        void init() override
        {
            auto cPtr = exchange->getCPtrCurrentSize();

            this->request = Environment<DIM>::get().EnvironmentController().getCommunicator().startSend(
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
                this->setFinished();
                return true;
            }
            return false;
        }

        ~TaskSendMPI() override
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return std::string("TaskSendMPI exchange type=") + std::to_string(exchange->getExchangeType());
        }

    private:
        Exchange<TYPE, DIM>* exchange;
        MPI_Request* request;
        MPI_Status status;
    };

} // namespace pmacc
