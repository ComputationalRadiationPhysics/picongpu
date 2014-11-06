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

#ifndef SIMULATIONFIELDHELPER_HPP
#define	SIMULATIONFIELDHELPER_HPP

#include "dimensions/GridLayout.hpp"
#include "eventSystem/EventSystem.hpp"
#include "mappings/kernel/MappingDescription.hpp"

namespace PMacc
{

template<class CellDescription>
class SimulationFieldHelper
{
public:

    typedef CellDescription MappingDesc;

    SimulationFieldHelper(CellDescription description) :
    cellDescription(description)
    {
    }
    /**
     * Reset is as well used for init.
     */
    virtual void reset(uint32_t currentStep) = 0;

    /**
     * Synchronize data from host to device.
     */
    virtual void syncToDevice() = 0;

    /**
     * Start an asynchronous communication.
     *
     * @param serialEvent an EventTask the method has to wait on
     * @return an EventTask which can be used to wait on completion
     */
    virtual EventTask asyncCommunication(EventTask serialEvent)=0;
protected:
    CellDescription cellDescription;
};

} //namespace PMacc

#endif	/* SIMULATIONFIELDHELPER_HPP */

