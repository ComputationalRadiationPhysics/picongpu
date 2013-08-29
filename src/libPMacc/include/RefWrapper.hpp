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
 
#ifndef REF_WRAPPER_HPP
#define REF_WRAPPER_HPP

template<typename Type>
class RefWrapper
{
private:
    Type& myRef;
public:
    typedef Type type;

    HDINLINE
    RefWrapper(Type& _ref) : myRef(_ref) {}
    
    HDINLINE
    RefWrapper(const RefWrapper<Type>& other) : myRef(other.myRef) {}
    
    HDINLINE
    void operator=(const Type& rhs)
    {
        this->myRef = rhs;
    }
    
    HDINLINE
    void operator=(const RefWrapper<Type>& rhs)
    {
        this->myRef = rhs.myRef;
    }
    
    HDINLINE
    operator Type&() const
    {
        return this->myRef;
    }
    
    HDINLINE
    Type& operator*() 
    {
        return this->myRef;
    }
    
    HDINLINE
    Type& operator()()
    {
        return this->myRef;
    }
        
    HDINLINE
    Type& get() const
    {
        return myRef;
    }
};

template<typename Type>
HDINLINE
RefWrapper<Type> ref(Type& _ref)
{
    return RefWrapper<Type>(_ref);
}

template<typename Type>
HDINLINE
RefWrapper<Type> byRef(Type& _ref)
{
    return RefWrapper<Type>(_ref);
}

#endif // REF_WRAPPER_HPP
