/**
 * Copyright 2013 Felix Schmitt, Rene Widera
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


#ifndef _TASKKERNEL_HPP
#define _TASKKERNEL_HPP

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "eventSystem/tasks/StreamTask.hpp"
#include "eventSystem/events/IEventData.hpp"

namespace PMacc
{
     
    class TaskKernel : public StreamTask
    {
    public:

        TaskKernel(std::string kernelName);

        virtual ~TaskKernel();

        bool executeIntern() throw (std::runtime_error);

        void event(id_t, EventType, IEventData*);

        void activateChecks();

        virtual std::string toString();

        virtual void init();

    private:
        bool canBeChecked;
        std::string kernelName;
    };

} //namespace PMacc


#endif	/* _TASKKERNEL_HPP */

