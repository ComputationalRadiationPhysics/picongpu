/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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
#include <boost/mpl/transform.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/type_traits.hpp>

#include <utility>


namespace pmacc
{
    namespace meta
    {
        namespace detail
        {
            /** call the functor were itBegin points to
             *
             *  \tparam itBegin iterator to an element in a mpl sequence
             *  \tparam itEnd iterator to the end of a mpl sequence
             *  \tparam isEnd true if itBegin == itEnd, else false
             */
            template<typename itBegin, typename itEnd, bool isEnd = boost::is_same<itBegin, itEnd>::value>
            struct CallFunctorOfIterator
            {
                typedef typename boost::mpl::next<itBegin>::type nextIt;
                typedef typename boost::mpl::deref<itBegin>::type Functor;
                typedef CallFunctorOfIterator<nextIt, itEnd> NextCall;

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

            /** Recursion end of ForEach */
            template<typename itBegin, typename itEnd>
            struct CallFunctorOfIterator<itBegin, itEnd, true>
            {
                PMACC_NO_NVCC_HDWARNING
                template<typename... T_Types>
                HDINLINE void operator()(T_Types&&...) const
                {
                }

                PMACC_NO_NVCC_HDWARNING
                template<typename... T_Types>
                HDINLINE void operator()(T_Types&&...)
                {
                }
            };

        } // namespace detail

        /** Compile-Time for each for Boost::MPL Type Lists
         *
         *  \tparam T_MPLSeq A mpl sequence that can be accessed by mpl::begin, mpl::end, mpl::next
         *  \tparam T_Functor An unary lambda functor with a HDINLINE void operator()(...) method
         *          _1 is substituted by Accessor's result using boost::mpl::apply with elements from T_MPLSeq.
         *          The maximum number of parameters for the operator() is limited by
         *          PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
         *  \tparam T_Accessor An unary lambda operation
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

            typedef typename bmpl::transform<T_MPLSeq, ReplacePlaceholder<bmpl::_1>>::type SolvedFunctors;

            typedef typename boost::mpl::begin<SolvedFunctors>::type begin;
            typedef typename boost::mpl::end<SolvedFunctors>::type end;


            typedef detail::CallFunctorOfIterator<begin, end> NextCall;

            /* this functor does nothing */
            typedef detail::CallFunctorOfIterator<end, end> Functor;

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

    } // namespace meta
} // namespace pmacc
