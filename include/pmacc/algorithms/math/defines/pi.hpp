/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/types.hpp"

#include <numbers>

namespace pmacc
{
    namespace math
    {
        /** Values of pi and related constants as T_Type
         */
        template<typename T_Type>
        struct Pi
        {
            static constexpr T_Type value = std::numbers::pi_v<T_Type>;
            static constexpr T_Type doubleValue = static_cast<T_Type>(2.0) * value;
            static constexpr T_Type halfValue = value / static_cast<T_Type>(2.0);
            static constexpr T_Type quarterValue = value / static_cast<T_Type>(4.0);
            static constexpr T_Type doubleReciprocalValue = static_cast<T_Type>(2.0) * std::numbers::inv_pi_v<T_Type>;
        };

    } // namespace math
} // namespace pmacc
