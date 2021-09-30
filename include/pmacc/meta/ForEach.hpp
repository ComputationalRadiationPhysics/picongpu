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

#include "pmacc/meta/SeqToList.hpp"
#include "pmacc/meta/accessors/Identity.hpp"

#include <boost/mpl/apply.hpp>

#include <type_traits>

namespace pmacc::meta
{
    /** Compile-Time for each for Boost::MPL Type Lists
     *
     *  @tparam T_MPLSeq A mpl sequence that can be accessed by mpl::begin, mpl::end, mpl::next
     *  @tparam T_Functor An unary lambda functor with a HDINLINE void operator()(...) method
     *          _1 is substituted by Accessor's result using boost::mpl::apply with elements from T_MPLSeq.
     *          The maximum number of parameters for the operator() is limited by
     *          PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     *  @tparam T_Accessor An unary lambda operation
     *
     * Example:
     *      MPLSeq = pmacc::mp_list<int,float>
     *      Functor = any unary lambda functor
     *      Accessor = lambda operation identity
     *
     *      definition: F(X) means boost::apply<F,X>
     *
     *      call:   ForEach<MPLSeq,Functor,Accessor>()(42);
     *      unrolled code: Functor(Accessor(int))(42);
     *                     Functor(Accessor(float))(42);
     */
    template<typename T_MPLSeq, typename T_Functor, typename T_Accessor = meta::accessors::Identity<>>
    struct ForEach
    {
        using List = detail::SeqToList<T_MPLSeq>;

        template<typename X>
        using ReplacePlaceholder =
            typename boost::mpl::apply1<T_Functor, typename boost::mpl::apply1<T_Accessor, X>::type>::type;

        using SolvedFunctors = mp_transform<ReplacePlaceholder, List>;

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
