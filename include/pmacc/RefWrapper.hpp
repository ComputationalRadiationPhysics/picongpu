/* Copyright 2013-2019 Heiko Burau, Rene Widera
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

namespace pmacc
{

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
    Type& get()
    {
        return myRef;
    }

        HDINLINE
    Type& get() const
    {
        return myRef;
    }
};

} // namespace pmacc
