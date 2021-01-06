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
#include "pmacc/random/Random.hpp"
#include "pmacc/Environment.hpp"

namespace pmacc
{
    namespace random
    {
        /**
         * A reference to a state of a RNG provider
         */
        template<class T_RNGProvider>
        struct RNGHandle
        {
            typedef T_RNGProvider RNGProvider;
            static constexpr uint32_t rngDim = RNGProvider::dim;
            typedef typename RNGProvider::DataBoxType RNGBox;
            typedef typename RNGProvider::RNGMethod RNGMethod;
            typedef typename RNGMethod::StateType RNGState;
            typedef pmacc::DataSpace<rngDim> RNGSpace;

            template<class T_Distribution>
            struct GetRandomType
            {
                typedef typename T_Distribution::template applyMethod<RNGMethod>::type Distribution;
                typedef Random<Distribution, RNGMethod, RNGState*> type;
            };

            /**
             * Creates an instance of the functor
             *
             * @param rngBox Databox of the RNG provider
             */
            RNGHandle(const RNGBox& rngBox) : m_rngBox(rngBox)
            {
            }

            /**
             * Initializes this instance
             *
             * \param cellIdx index into the underlying RNG provider
             */
            HDINLINE void init(const RNGSpace& cellIdx)
            {
                m_rngBox = m_rngBox.shift(cellIdx);
            }

            HDINLINE RNGState& getState()
            {
                return m_rngBox(RNGSpace::create(0));
            }

            HDINLINE RNGState& operator*()
            {
                return m_rngBox(RNGSpace::create(0));
            }

            HDINLINE RNGState& operator->()
            {
                return m_rngBox(RNGSpace::create(0));
            }

            template<class T_Distribution>
            HDINLINE typename GetRandomType<T_Distribution>::type applyDistribution()
            {
                return typename GetRandomType<T_Distribution>::type(&getState());
            }

        protected:
            PMACC_ALIGN8(m_rngBox, RNGBox);
        };

    } // namespace random
} // namespace pmacc
