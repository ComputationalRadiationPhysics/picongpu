/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2015-2024 Richard Pausch
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

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
            operator()(T_Functor const diffEq, T_Variable const var, T_Time const time, T_Time const deltaTime)
            {
                // use typenames instead of template types
                using FunctorType = T_Functor;
                using VariableType = T_Variable;
                using TimeType = T_Time;

                // calculate all 4 steps of the Runge Kutta 4th order
                VariableType const k_1 = diffEq(time, var);
                VariableType const k_2
                    = diffEq(time + TimeType(0.5) * deltaTime, var + (TimeType(0.5) * deltaTime) * k_1);
                VariableType const k_3
                    = diffEq(time + TimeType(0.5) * deltaTime, var + (TimeType(0.5) * deltaTime) * k_2);
                VariableType const k_4 = diffEq(time + deltaTime, var + deltaTime * k_3);

                // combine all 4 steps
                VariableType const diff
                    = deltaTime / TimeType(6.) * (k_1 + TimeType(2.) * k_2 + TimeType(2.) * k_3 + k_4);

                // current var + difference = new var
                VariableType const out = var + diff;
                return out;
            }
        };


    } // namespace math
} // namespace pmacc
