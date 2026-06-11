/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/RNGState.hpp"
#include "pmacc/types.hpp"

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
            using RNGMethod = T_RNGMethod;
            /* RNGHandle assumed */
            using RNGHandle = T_RNGStatePtrOrHandle;
            using Distribution = T_Distribution;
            using result_type = typename Distribution::result_type;

            /** This can be constructed with either the RNGBox (like the RNGHandle) or from an RNGHandle instance */
            template<class T_RNGBoxOrHandle>
            HINLINE explicit Random(T_RNGBoxOrHandle const& rngBox) : RNGHandle(rngBox)
            {
            }

            /**
             * Initializes this instance
             *
             * @param cellIdx index into the underlying RNG Provider
             */
            template<typename T_Offset>
            HDINLINE void init(T_Offset const& cellIdx)
            {
                RNGHandle::init(cellIdx);
            }

            /** Returns a new random number advancing the state */
            template<typename T_Worker>
            DINLINE result_type operator()(T_Worker const& worker)
            {
                return Distribution::operator()(worker, RNGHandle::getState());
            }
        };

        /**
         * Specialization when the state is a pointer
         */
        template<class T_Distribution, class T_RNGMethod, class T_RNGState>
        struct Random<T_Distribution, T_RNGMethod, T_RNGState*> : private T_Distribution
        {
            using RNGMethod = T_RNGMethod;
            using RNGState = T_RNGState;
            using Distribution = T_Distribution;
            using result_type = typename Distribution::result_type;

            HDINLINE Random() : m_rngState(nullptr)
            {
            }

            HDINLINE Random(RNGState* m_rngState) : m_rngState(m_rngState)
            {
            }

            /** Returns a new random number advancing the state */
            template<typename T_Worker>
            DINLINE result_type operator()(T_Worker const& worker)
            {
                return Distribution::operator()(worker, *m_rngState);
            }

        protected:
            PMACC_ALIGN(m_rngState, RNGState*);
        };

    } // namespace random
} // namespace pmacc
