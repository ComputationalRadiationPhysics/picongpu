/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Alexander Debus
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
    namespace math
    {
        template<typename ArgType, typename SinType, typename CosType>
        struct SinCos;

        template<typename Type>
        struct Sinc;


        template<typename ArgType, typename SinType, typename CosType>
        HDINLINE typename SinCos<ArgType, SinType, CosType>::result sincos(
            ArgType arg,
            SinType& sinValue,
            CosType& cosValue)
        {
            return SinCos<ArgType, SinType, CosType>()(arg, sinValue, cosValue);
        }

        template<typename T1>
        HDINLINE typename Sinc<T1>::result sinc(const T1& value)
        {
            return Sinc<T1>()(value);
        }

    } /* namespace math */
} /* namespace pmacc */
