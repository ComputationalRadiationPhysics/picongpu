/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace traits
    {
        /** Get number of possible exchanges
         *
         * @tparam T_dim dimension of the simulation
         * @return \p ::value number of possible exchanges
         *              (is number neighbors + myself)
         */
        template<uint32_t T_dim>
        struct NumberOfExchanges;

        template<>
        struct NumberOfExchanges<DIM1>
        {
            static constexpr uint32_t value = LEFT + RIGHT;
        };

        template<>
        struct NumberOfExchanges<DIM2>
        {
            static constexpr uint32_t value = TOP + BOTTOM;
        };

        template<>
        struct NumberOfExchanges<DIM3>
        {
            static constexpr uint32_t value = BACK + FRONT;
        };

    } // namespace traits

} // namespace pmacc
