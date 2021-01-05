/* Copyright 2015-2021 Richard Pausch
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

namespace pmacc
{
    namespace math
    {
        struct RungeKutta4
        {
            /** Runge Kutta solver 4th order
             *
             *  Calculate next time step based on the Runge Kutta
             *  algorithm and return next variable
             *
             *  @param diffEq functor with first argument time and second variables
             *  @param var variables of type T_Variable (can be vector type)
             *  @param time current time
             *  @param deltaTime time step
             *  @return var for the consecutive time step
             */
            template<typename T_Functor, typename T_Variable, typename T_Time>
            HDINLINE T_Variable
            operator()(const T_Functor diffEq, const T_Variable var, const T_Time time, const T_Time deltaTime)
            {
                // use typenames instead of template types
                typedef T_Functor FunctorType;
                typedef T_Variable VariableType;
                typedef T_Time TimeType;

                // calculate all 4 steps of the Runge Kutta 4th order
                const VariableType k_1 = diffEq(time, var);
                const VariableType k_2
                    = diffEq(time + TimeType(0.5) * deltaTime, var + (TimeType(0.5) * deltaTime) * k_1);
                const VariableType k_3
                    = diffEq(time + TimeType(0.5) * deltaTime, var + (TimeType(0.5) * deltaTime) * k_2);
                const VariableType k_4 = diffEq(time + deltaTime, var + deltaTime * k_3);

                // combine all 4 steps
                const VariableType diff
                    = deltaTime / TimeType(6.) * (k_1 + TimeType(2.) * k_2 + TimeType(2.) * k_3 + k_4);

                // current var + difference = new var
                const VariableType out = var + diff;
                return out;
            }
        };


    } // namespace math
} // namespace pmacc
