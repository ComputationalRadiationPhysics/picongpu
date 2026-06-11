/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/random/Random.hpp"
#include "pmacc/types.hpp"

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
            using RNGProvider = T_RNGProvider;
            static constexpr uint32_t rngDim = RNGProvider::dim;
            using RNGBox = typename RNGProvider::DataBoxType;
            using RNGMethod = typename RNGProvider::RNGMethod;
            using RNGState = typename RNGMethod::StateType;
            using RNGSpace = pmacc::DataSpace<rngDim>;

            template<class T_Distribution>
            struct GetRandomType
            {
                using Distribution = typename T_Distribution::template applyMethod<RNGMethod>::type;
                using type = Random<Distribution, RNGMethod, RNGState*>;
            };

            /**
             * Creates an instance of the functor
             *
             * @param rngBox Databox of the RNG provider
             */
            RNGHandle(RNGBox const& rngBox) : m_rngBox(rngBox)
            {
            }

            /**
             * Initializes this instance
             *
             * @param cellIdx index into the underlying RNG provider
             */
            HDINLINE void init(RNGSpace const& cellIdx)
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
            PMACC_ALIGN(m_rngBox, RNGBox);
        };

    } // namespace random
} // namespace pmacc
