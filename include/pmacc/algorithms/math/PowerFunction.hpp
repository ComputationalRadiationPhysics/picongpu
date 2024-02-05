/* Copyright 2022-2023 Brian Marre, Rene Widera, Richard Pausch
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
#ifndef NDEBUG
#    include "pmacc/static_assert.hpp"
#endif

#include <cstdint>
#include <type_traits>

namespace pmacc::math
{
    /** power function for integer exponents, constexpr
     *
     * @tparam T_Type return and accumulation data type
     * @tparam T_Exp exponent data type, must be an unsigned integral type, default uint32_t
     *
     * @param x base
     * @param exp exponent
     */
    template<typename T_Type, typename T_Exp = uint32_t>
    HDINLINE constexpr T_Type cPow(T_Type const x, T_Exp const exp)
    {
        static_assert(
            std::is_unsigned_v<T_Exp>,
            "This pow() implementation supports only unsigned integral types for the exponent.");

        T_Type result = static_cast<T_Type>(1u);
        for(T_Exp e = static_cast<T_Exp>(0u); e < exp; e++)
            // for whatever reason "*=" causes the function to return wrong results, do not ask me ...
            result = result * x;
        return result;
    }

    namespace test
    {
#ifndef NDEBUG
        PMACC_CASSERT_MSG(
            FAIL_unitTest_2_power_0,
            cPow(static_cast<uint32_t>(2u), static_cast<uint32_t>(0u)) == static_cast<uint32_t>(1u));
        PMACC_CASSERT_MSG(
            FAIL_unitTest_2_power_1,
            cPow(static_cast<uint8_t>(2u), static_cast<uint8_t>(1u)) == static_cast<uint8_t>(2u));
        PMACC_CASSERT_MSG(
            FAIL_unitTest_4_power_4,
            cPow(static_cast<uint32_t>(4u), static_cast<uint8_t>(4u)) == static_cast<uint32_t>(256u));
        PMACC_CASSERT_MSG(FAIL_unitTest_2_power_2, cPow(2., static_cast<uint8_t>(2u)) == 4.);
#endif
    } // namespace test

} // namespace pmacc::math
