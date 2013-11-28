/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
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
 

#ifndef DEBUGDATASPACE_HPP
#define	DEBUGDATASPACE_HPP

#include <string>
#include <sstream>

#include "dimensions/DataSpace.hpp"

namespace PMacc
{

    /**
     * Helper class for debugging DataSpaces
     * 
     * @tparam DIM dimension of the DataSpace to debug.
     */
    template <unsigned DIM>
    class DebugDataSpace
    {
    public:
        static std::string dspToStr(DataSpace<DIM>& dsp);
    };
    
    template <>
    class DebugDataSpace<DIM2>
    {
    public:
        static std::string dspToStr(DataSpace<DIM2>& dsp)
        {
            std::stringstream stream;
            
            stream << "(" << dsp.x() << ", " << dsp.y() << ")";
            
            return stream.str();
        }
    };
    
    template <>
    class DebugDataSpace<DIM3>
    {
    public:
        static std::string dspToStr(DataSpace<DIM3>& dsp)
        {
            std::stringstream stream;
            
            stream << "(" << dsp.x() << ", " << dsp.y() << ", " << dsp.z() << ")";
            
            return stream.str();
        }
    };
    
}

#endif	/* DEBUGDATASPACE_HPP */

