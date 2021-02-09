/* Copyright 2015-2021 Alexander Grund
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
#include "pmacc/random/RNGState.hpp"
#include <boost/utility/result_of.hpp>

namespace pmacc
{
    namespace random
    {
        /**
         * Random Number Generator. Functor that returns a random number per call
         *
         * Default implementation assumes a RNGHandle
         */
        template<
            class T_Distribution,
            class T_RNGMethod,
            class T_RNGStatePtrOrHandle = typename T_RNGMethod::StateType*>
        struct Random
            : private T_Distribution
            , private T_RNGStatePtrOrHandle
        {
            typedef T_RNGMethod RNGMethod;
            /* RNGHandle assumed */
            typedef T_RNGStatePtrOrHandle RNGHandle;
            typedef T_Distribution Distribution;
            typedef typename boost::result_of<Distribution(typename RNGHandle::RNGState&)>::type result_type;

            /** This can be constructed with either the RNGBox (like the RNGHandle) or from an RNGHandle instance */
            template<class T_RNGBoxOrHandle>
            explicit HINLINE Random(const T_RNGBoxOrHandle& rngBox) : RNGHandle(rngBox)
            {
            }

            /**
             * Initializes this instance
             *
             * \param cellIdx index into the underlying RNG Provider
             */
            template<typename T_Offset>
            HDINLINE void init(const T_Offset& cellIdx)
            {
                RNGHandle::init(cellIdx);
            }

            /** Returns a new random number advancing the state */
            template<typename T_Acc>
            DINLINE result_type operator()(T_Acc const& acc)
            {
                return Distribution::operator()(acc, RNGHandle::getState());
            }
        };

        /**
         * Specialization when the state is a pointer
         */
        template<class T_Distribution, class T_RNGMethod, class T_RNGState>
        struct Random<T_Distribution, T_RNGMethod, T_RNGState*> : private T_Distribution
        {
            typedef T_RNGMethod RNGMethod;
            typedef T_RNGState RNGState;
            typedef T_Distribution Distribution;
            typedef typename boost::result_of<Distribution(RNGState&)>::type result_type;

            HDINLINE Random() : m_rngState(nullptr)
            {
            }

            HDINLINE Random(RNGState* m_rngState) : m_rngState(m_rngState)
            {
            }

            /** Returns a new random number advancing the state */
            template<typename T_Acc>
            DINLINE result_type operator()(T_Acc const& acc)
            {
                return Distribution::operator()(acc, *m_rngState);
            }

        protected:
            PMACC_ALIGN(m_rngState, RNGState*);
        };

    } // namespace random
} // namespace pmacc
