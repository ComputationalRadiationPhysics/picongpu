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
 
#ifndef TYPES_HPP
#define	TYPES_HPP

#include "types.hpp"
#include "dimensions/DataSpace.hpp"
#include "memory/buffers/GridBuffer.hpp"

namespace gol
{
    using namespace PMacc;

    typedef DataSpace<DIM2> Space;
    typedef GridController<DIM2> GC;
    typedef GridBuffer<uint8_t, DIM2 > Buffer;

    enum CommunicationTags
    {
        BUFF1 = 0u, BUFF2 = 1u
    };
}

#endif	/* TYPES_HPP */

