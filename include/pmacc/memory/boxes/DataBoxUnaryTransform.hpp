/* Copyright 2014-2021 Rene Widera
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
    /** DataBox which apply a unary functor on every operator () and [] access
     *
     * @tparam T_Base base class to inherit from
     * @tparam T_UnaryFunctor unary functor which is applied on every access
     *         - template parameter of functor is the input type for the functor
     *         - functor must have defined the result type as ::result
     */
    template<class T_Base, template<typename> class T_UnaryFunctor>
    class DataBoxUnaryTransform : public T_Base
    {
    public:
        typedef T_Base Base;
        typedef typename Base::ValueType BaseValueType;

        typedef T_UnaryFunctor<BaseValueType> UnaryFunctor;

        typedef typename UnaryFunctor::result ValueType;
        typedef ValueType RefValueType;
        static constexpr uint32_t Dim = Base::Dim;

        HDINLINE DataBoxUnaryTransform(const Base& base) : Base(base)
        {
        }

        HDINLINE DataBoxUnaryTransform() : Base()
        {
        }

        template<typename T_Index>
        HDINLINE ValueType operator()(const T_Index& idx) const
        {
            return UnaryFunctor()(Base::operator()(idx));
        }

        template<typename T_Index>
        HDINLINE ValueType operator()(const T_Index& idx)
        {
            return UnaryFunctor()(Base::operator()(idx));
        }

        template<typename T_Index>
        HDINLINE ValueType operator[](const T_Index idx)
        {
            return UnaryFunctor()(Base::operator[](idx));
        }

        template<typename T_Index>
        HDINLINE ValueType operator[](const T_Index idx) const
        {
            return UnaryFunctor()(Base::operator[](idx));
        }
    };

} // namespace pmacc
