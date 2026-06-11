/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        /**
         * Wrapper class for a state of a random number generator
         * Can be used for aligned storing of states
         */
        template<class T_RNGMethod>
        class RNGState
        {
        public:
            using RNGMethod = T_RNGMethod;
            using StateType = typename RNGMethod::StateType;

            HDINLINE RNGState() = default;

            HDINLINE RNGState(StateType const& other) : state(other)
            {
            }

            HDINLINE StateType& getState()
            {
                return state;
            }

        private:
            PMACC_ALIGN8(StateType, ) state;
        };

    } // namespace random
} // namespace pmacc
