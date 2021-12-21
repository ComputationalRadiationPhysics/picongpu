/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace alpaka
{
    namespace test
    {
        namespace unit
        {
            namespace math
            {
                // New types need to be added to the switch-case in DataGen.hpp
                enum class Range
                {
                    OneNeighbourhood,
                    PositiveOnly,
                    PositiveAndZero,
                    NotZero,
                    Unrestricted
                };

                // New types need to be added to the operator() function in Functor.hpp
                enum class Arity
                {
                    Unary = 1,
                    Binary = 2
                };

                template<typename T, Arity Tarity>
                struct ArgsItem
                {
                    static constexpr Arity arity = Tarity;
                    static constexpr size_t arity_nr = static_cast<size_t>(Tarity);

                    T arg[arity_nr]; // represents arg0, arg1, ...

                    friend std::ostream& operator<<(std::ostream& os, const ArgsItem& argsItem)
                    {
                        os.precision(17);
                        os << "[ ";
                        for(size_t i = 0; i < argsItem.arity_nr; ++i)
                            os << std::setprecision(std::numeric_limits<T>::digits10 + 1) << argsItem.arg[i] << ", ";
                        os << "]";
                        return os;
                    }
                };

                template<typename T>
                auto rsqrt(T const& arg) -> decltype(std::sqrt(arg))
                {
                    return static_cast<T>(1) / std::sqrt(arg);
                }

            } // namespace math
        } // namespace unit
    } // namespace test
} // namespace alpaka
