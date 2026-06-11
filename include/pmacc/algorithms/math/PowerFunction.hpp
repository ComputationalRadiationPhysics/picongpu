/*
 * SPDX-FileCopyrightText: Brian Marre, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once


#include "pmacc/types.hpp"
#ifndef NDEBUG
#    include "pmacc/static_assert.hpp"
#endif

#include <concepts>
#include <cstdint>

namespace pmacc::math
{
    /** power function for non negative integer exponents, constexpr
     *
     * @tparam T_Type return and accumulation data type
     * @tparam T_Exp exponent data type, must be an unsigned integral type, default uint32_t
     *
     * @param x base
     * @param exp exponent
     */
    template<typename T_Type, std::unsigned_integral T_Exp = uint32_t>
    HDINLINE constexpr T_Type cPow(T_Type base, T_Exp exp) noexcept
    {
        T_Type result{1};
        while(exp > 0)
        {
            if(exp & 1)
            {
                result *= base;
            }

            base *= base;
            exp >>= 1;
        }
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
