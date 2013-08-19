/**
 * Copyright 2013 René Widera
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
 
#ifndef ABSTRACTINITIALISER_HPP
#define	ABSTRACTINITIALISER_HPP

#include "dataManagement/ISimulationData.hpp"
#include "dataManagement/IDataSorter.hpp"


namespace PMacc
{

    /**
     * Abstract base class for initialising simulation data (ISimulationData).
     */
    class AbstractInitialiser
    {
    public:
        /**
         * Setup this initialiser.
         * Called before any init.
         * 
         * @return the next timestep
         */
        virtual uint32_t setup() { return 0;};
        
        /**
         * Tears down this initialiser.
         * Called after any init.
         */
        virtual void teardown() {};
        
        /**
         * Initialises simulation data (concrete type of data is described by id).
         * 
         * @param id identifier for simulation data (identifies type, too)
         * @param data reference to actual simulation data
         */
        virtual void init(uint32_t id, ISimulationData& data,uint32_t currentStep) = 0;
    };
    
}

#endif	/* ABSTRACTINITIALISER_HPP */

