/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
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

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{
    namespace private_Box
    {
        template<unsigned DIM, class Base>
        class Box;

        template<class Base>
        class Box<DIM1, Base> : public Base
        {
        public:
            enum
            {
                Dim = DIM1
            };
            typedef typename Base::ValueType ValueType;
            typedef typename Base::RefValueType RefValueType;

            HDINLINE RefValueType operator()(const DataSpace<DIM1>& idx = DataSpace<DIM1>()) const
            {
                return Base::operator[](idx.x());
            }

            HDINLINE RefValueType operator()(const DataSpace<DIM1>& idx = DataSpace<DIM1>())
            {
                return Base::operator[](idx.x());
            }

            HDINLINE Box(Base base) : Base(base)
            {
            }

            HDINLINE Box() : Base()
            {
            }
        };

        template<class Base>
        class Box<DIM2, Base> : public Base
        {
        public:
            enum
            {
                Dim = DIM2
            };
            typedef typename Base::ValueType ValueType;
            typedef typename Base::RefValueType RefValueType;

            HDINLINE RefValueType operator()(const DataSpace<DIM2>& idx = DataSpace<DIM2>()) const
            {
                return (Base::operator[](idx.y()))[idx.x()];
            }

            HDINLINE RefValueType operator()(const DataSpace<DIM2>& idx = DataSpace<DIM2>())
            {
                return (Base::operator[](idx.y()))[idx.x()];
            }

            HDINLINE Box(Base base) : Base(base)
            {
            }

            HDINLINE Box() : Base()
            {
            }
        };

        template<class Base>
        class Box<DIM3, Base> : public Base
        {
        public:
            enum
            {
                Dim = DIM3
            };
            typedef typename Base::ValueType ValueType;
            typedef typename Base::RefValueType RefValueType;

            HDINLINE RefValueType operator()(const DataSpace<DIM3>& idx = DataSpace<DIM3>()) const
            {
                return (Base::operator[](idx.z()))[idx.y()][idx.x()];
            }

            HDINLINE RefValueType operator()(const DataSpace<DIM3>& idx = DataSpace<DIM3>())
            {
                return (Base::operator[](idx.z()))[idx.y()][idx.x()];
            }

            HDINLINE Box(Base base) : Base(base)
            {
            }

            HDINLINE Box() : Base()
            {
            }
        };


    } // namespace private_Box

    template<class Base>
    class DataBox : public private_Box::Box<Base::Dim, Base>
    {
    public:
        typedef typename Base::ValueType ValueType;
        typedef DataBox<Base> Type;
        typedef typename Base::RefValueType RefValueType;

        HDINLINE DataBox(Base base) : private_Box::Box<Base::Dim, Base>(base)
        {
        }

        HDINLINE DataBox() : private_Box::Box<Base::Dim, Base>()
        {
        }

        HDINLINE Type shift(const DataSpace<Base::Dim>& offset) const
        {
            Type result(*this);
            result.fixedPointer = &((*this)(offset));
            return result;
        }

        HDINLINE DataBox<typename Base::ReducedType> reduceZ(const int zOffset) const
        {
            return DataBox<typename Base::ReducedType>(Base::reduceZ(zOffset));
        }
    };

} // namespace pmacc
