/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            namespace util
            {
                // goal: to increase readability of code

                template<typename A> /// a generic square function
                HDINLINE A square(A a)
                {
                    return a * a;
                }

                template<typename A, typename R> /// a more generic square function
                HDINLINE R square(A a)
                {
                    return a * a;
                }

                template<typename A> /// a generic cube function
                HDINLINE A cube(A a)
                {
                    return a * a * a;
                }

                template<typename A, typename R> /// a more generic cube function
                HDINLINE R cube(A a)
                {
                    return a * a * a;
                }

                template<typename A, typename R = A> /// a more generic square struct
                struct Cube
                {
                    HDINLINE R operator()(A a)
                    {
                        return a * a * a;
                    }
                };

                template<typename A, typename R = A> /// a more generic square struct
                struct Square
                {
                    HDINLINE R operator()(A a) const
                    {
                        return a * a;
                    }
                };


                namespace details
                {
                    /** power function - with extra const parameter for efficient code
                     *
                     * T_type requires cast from int and multiplication
                     * @tparam T_Type - base type
                     * @param x - base value
                     * @param exp - exponent
                     * @param results (=1) - do not change - workaround to produce efficient code
                     * @return std::pow(x, exp)
                     */
                    template<typename T_Type>
                    HDINLINE constexpr T_Type pow(T_Type const x, uint32_t const exp, const T_Type result = T_Type(1))
                    {
                        return exp == 0 ? result
                                        : (exp == 1 ? x * result : util::details::pow(x, exp - 1, result * x));
                    }
                } // namespace details

                /** power function
                 *
                 * T_type requires cast from int and multiplication
                 * @tparam T_Type - base type
                 * @param x - base value
                 * @param exp - exponent
                 * @return std::pow(x, exp)
                 */
                template<typename T_Type>
                HDINLINE constexpr T_Type pow(T_Type const x, uint32_t const exp)
                {
                    return util::details::pow(x, exp);
                }

            } // namespace util

        } // namespace radiation

    } // namespace plugins

} // namespace picongpu
