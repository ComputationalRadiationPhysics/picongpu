/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

#pragma once

#include "pmacc/communication/manager_common.hpp"
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"

#include <mpi.h>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class TaskReceiveMPI : public MPITask
    {
    public:
        TaskReceiveMPI(Exchange<TYPE, DIM>* exchange) : MPITask(), exchange(exchange)
        {
        }

        virtual void init()
        {
            Buffer<TYPE, DIM>* dst = exchange->getCommunicationBuffer();

            this->request = Environment<DIM>::get().EnvironmentController().getCommunicator().startReceive(
                exchange->getExchangeType(),
                reinterpret_cast<char*>(dst->getPointer()),
                dst->getDataSpace().productOfComponents() * sizeof(TYPE),
                exchange->getCommunicationTag());
        }

        bool executeIntern()
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

        virtual ~TaskReceiveMPI()
        {
            //! \todo this make problems because we send bytes and not combined types
            int recv_data_count;
            MPI_CHECK_NO_EXCEPT(MPI_Get_count(&(this->status), MPI_CHAR, &recv_data_count));


            IEventData* edata = new EventDataReceive(nullptr, recv_data_count);

            notify(this->myId, RECVFINISHED, edata); /*add notify her*/
            __delete(edata);
        }

        void event(id_t, EventType, IEventData*)
        {
        }

        std::string toString()
        {
            return std::string("TaskReceiveMPI exchange type=") + std::to_string(exchange->getExchangeType());
        }

    private:
        Exchange<TYPE, DIM>* exchange;
        MPI_Request* request;
        MPI_Status status;
    };

} // namespace pmacc
