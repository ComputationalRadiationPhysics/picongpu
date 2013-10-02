/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
#pragma once

namespace PMacc
{
namespace container
{

template<typename Buffer>
struct View : public Buffer
{
    HDINLINE View() {}
    template<typename TBuffer>
    HDINLINE View(const View<TBuffer>& other)
    {
        *this = other;
    }
    
    HDINLINE ~View() 
    {
        (*this->refCount)++;
    }
    
    template<typename TBuffer>
    HDINLINE View& operator=(const View<TBuffer>& other)
    {
        this->dataPointer = other.dataPointer;
        this->_size = other._size;
        this->pitch = other.pitch;
        this->refCount = other.refCount;
        
        return *this;
    }

};
    
    
} // container
} // PMacc
