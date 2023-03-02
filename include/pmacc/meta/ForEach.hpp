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

#include "pmacc/meta/accessors/Identity.hpp"

#include <boost/mpl/apply.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/transform.hpp>

#include <type_traits>
#include <utility>


namespace pmacc::meta
{
    namespace detail
    {
        /** call the functor were itBegin points to
         *
         *  @tparam itBegin iterator to an element in a mpl sequence
         *  @tparam itEnd iterator to the end of a mpl sequence
         */
        template<typename itBegin, typename itEnd>
        struct CallFunctorOfIterator
        {
            using Functor = typename boost::mpl::deref<itBegin>::type;

            PMACC_NO_NVCC_HDWARNING
            template<typename... T_Types>
            HDINLINE void operator()(T_Types&&... ts) const
            {
                if constexpr(!std::is_same_v<itBegin, itEnd>)
                {
                    Functor()(std::forward<T_Types>(ts)...);
                    using NextCall = CallFunctorOfIterator<typename boost::mpl::next<itBegin>::type, itEnd>;
                    NextCall()(ts...);
                }
            }

            PMACC_NO_NVCC_HDWARNING
            template<typename... T_Types>
            HDINLINE void operator()(T_Types&&... ts)
            {
                if constexpr(!std::is_same_v<itBegin, itEnd>)
                {
                    Functor()(std::forward<T_Types>(ts)...);
                    using NextCall = CallFunctorOfIterator<typename boost::mpl::next<itBegin>::type, itEnd>;
                    NextCall()(ts...);
                }
            }
        };
    } // namespace detail

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
     *      MPLSeq = boost::mpl::vector<int,float>
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
        template<typename X>
        struct ReplacePlaceholder : bmpl::apply1<T_Functor, typename bmpl::apply1<T_Accessor, X>::type>
        {
        };

        using SolvedFunctors = typename bmpl::transform<T_MPLSeq, ReplacePlaceholder<bmpl::_1>>::type;

        using begin = typename boost::mpl::begin<SolvedFunctors>::type;
        using end = typename boost::mpl::end<SolvedFunctors>::type;


        using NextCall = detail::CallFunctorOfIterator<begin, end>;

        /* this functor does nothing */
        using Functor = detail::CallFunctorOfIterator<end, end>;

        PMACC_NO_NVCC_HDWARNING
        template<typename... T_Types>
        HDINLINE void operator()(T_Types&&... ts) const
        {
            Functor()(std::forward<T_Types>(ts)...);
            NextCall()(ts...);
        }

        PMACC_NO_NVCC_HDWARNING
        template<typename... T_Types>
        HDINLINE void operator()(T_Types&&... ts)
        {
            Functor()(std::forward<T_Types>(ts)...);
            NextCall()(ts...);
        }
    };

} // namespace pmacc::meta
