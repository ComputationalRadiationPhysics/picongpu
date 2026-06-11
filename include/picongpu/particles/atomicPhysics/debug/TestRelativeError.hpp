/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::debug
{
    /** test for relative error limit
     *
     * @tparam T_Type data type of quantity
     * @tparam T_consoleOutput true =^= write output also to console, false =^= no console output
     *
     * @param correctValue expected value
     * @param testValue actual value
     * @param descriptionQuantity short description of quantity tested
     * @param errorLimit maximm accepted relative error
     *
     * @attention may only be executed serially on cpu
     *
     * @return true =^= SUCCESS, false =^= FAIL
     */
    template<bool T_consoleOutput, typename T_Type>
    static bool testRelativeError(
        T_Type const correctValue,
        T_Type const testValue,
        std::string const descriptionQuantity = "",
        T_Type const errorLimit = static_cast<T_Type>(1e-9))
    {
        T_Type const relativeError = math::abs(testValue / correctValue - static_cast<T_Type>(1.f));

        if constexpr(T_consoleOutput)
        {
            std::cout << "[relative error] " << descriptionQuantity << ":\t" << relativeError << std::endl;
            std::cout << "\t is:        " << testValue << std::endl;
            std::cout << "\t should be: " << correctValue << std::endl;
        }

        if(std::isnan(relativeError))
        {
            // FAIL
            if constexpr(T_consoleOutput)
                std::cout << "\t x FAIL, is NaN" << std::endl;
            return false;
        }

        if(relativeError > errorLimit)
        {
            // FAIL
            if constexpr(T_consoleOutput)
                std::cout << "\t x FAIL, > errorLimit" << std::endl;
            return false;
        }
        else
        {
            // SUCCESS
            if constexpr(T_consoleOutput)
                std::cout << "\t * SUCCESS" << std::endl;
            return true;
        }
    }
} // namespace picongpu::particles::atomicPhysics::debug
