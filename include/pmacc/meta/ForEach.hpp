/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/meta/accessors/Identity.hpp"

#include <boost/mpl/apply.hpp>

#include <type_traits>

namespace pmacc::meta
{
    /** Compile-Time for each for Boost::MPL Type Lists
     *
     *  @tparam List An mp_list.
     *  @tparam T_Functor An unary lambda functor with a HDINLINE void operator()(...) method
     *          _1 is substituted by Accessor's result using boost::mpl::apply with elements from T_MPLSeq.
     *          The maximum number of parameters for the operator() is limited by
     *          PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     *  @tparam T_Accessor An unary lambda operation
     *
     * Example:
     *      List = pmacc::mp_list<int,float>
     *      Functor = any unary lambda functor
     *      Accessor = lambda operation identity
     *
     *      definition: F(X) means boost::apply<F,X>
     *
     *      call:   ForEach<List,Functor,Accessor>()(42);
     *      unrolled code: Functor(Accessor(int))(42);
     *                     Functor(Accessor(float))(42);
     */
    template<typename List, typename T_Functor, typename T_Accessor = meta::accessors::Identity<>>
    struct ForEach
    {
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
