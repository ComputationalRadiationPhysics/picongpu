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
 * File:   ISimulationData.hpp
 * Author: fschmitt
 *
 * Created on 21. Januar 2011, 11:27
 */

#ifndef ISIMULATIONDATA_HPP
#define	ISIMULATIONDATA_HPP

namespace PMacc
{
    /**
     * Interface for simulation data which should be registered at DataConnector
     * for file output, visualization, etc.
     */
    class ISimulationData
    {
    public:
        /**
         * Synchronizes simulation data, meaning accessing (host side) data
         * will return up-to-date values.
         */
        virtual void synchronize() = 0;
       
    };
}

#endif	/* ISIMULATIONDATA_HPP */

