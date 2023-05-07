/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "Apply.hpp"
#include "Mp11.hpp"

#include <type_traits>

namespace pmacc::meta
{
    /** Compile-Time for each for type lists
     *
     *  @tparam List An mp_list.
     *  @tparam T_Functor An unary lambda functor with a HDINLINE void operator()(...) method
     *          _1 is substituted by Accessor's result using Apply with elements from List.
     *  @tparam T_Accessor An unary lambda operation
     *
     * Example:
     *      List = pmacc::mp_list<int,float>
     *      Functor = any unary lambda functor
     *      Accessor = lambda operation identity
     *
     *      call:   ForEach<List,Functor,Accessor>()(42);
     *      unrolled code: Functor(Accessor(int))(42);
     *                     Functor(Accessor(float))(42);
     */
    template<typename List, typename T_Functor, typename T_Accessor = mp_identity<_1>>
    struct ForEach
    {
        template<typename T>
        using MakeFunctor = Apply<T_Functor, typename Apply<T_Accessor, T>::type>;

        using SolvedFunctors = mp_transform<MakeFunctor, List>;

        template<typename... T_Types>
        HDINLINE void operator()(T_Types&&... ts) const
        {
            callEachFunctorWithArgs(SolvedFunctors{}, std::forward<T_Types>(ts)...);
        }

    private:
        PMACC_NO_NVCC_HDWARNING
        template<typename... TFunctors, typename... TArgs>
        HDINLINE void callEachFunctorWithArgs(mp_list<TFunctors...>, TArgs&&... args) const
        {
            (TFunctors{}(std::forward<TArgs>(args)...), ...);
        }
    };
} // namespace pmacc::meta
