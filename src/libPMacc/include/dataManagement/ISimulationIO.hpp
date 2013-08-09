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
 
/* 
 * File:   ISimulationIO.hpp
 * Author: fschmitt
 *
 * Created on 19. Januar 2011, 16:36
 */

#ifndef ISIMULATIONIO_HPP
#define	ISIMULATIONIO_HPP

namespace PMacc
{

    /**
     * Interface for simulation data IO handlers.
     *
     * Should be implemented by classes which register at DataConnector
     * and handle simulation data IO.
     */
    class ISimulationIO
    {
    public:
        /**
         * Is called by DataConnector when data should be dumped.
         *
         * @param currentStep current simulation iteration step
         */
        virtual void notify(uint32_t currentStep) = 0;

        virtual ~ISimulationIO()
        {
        }
    protected:
    };
}

#endif	/* ISIMULATIONIO_HPP */

