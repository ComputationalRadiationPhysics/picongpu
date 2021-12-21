/* Copyright 2017-2021 Rene Widera
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
    namespace filter
    {
        namespace operators
        {
            //! combine all arguments by AND `&&`
            struct And
            {
                /** return a
                 *
                 * @param a a boolean value
                 * @return the input argument
                 */
                template<typename T_Arg>
                HDINLINE bool operator()(T_Arg const a) const
                {
                    return a;
                }

                /** get AND combined result
                 *
                 * @param args arguments to combine
                 * @return AND combination of all arguments
                 */
                template<typename T_Arg1, typename... T_Args>
                HDINLINE bool operator()(T_Arg1 const a, T_Args const... args) const
                {
                    return a && And{}(args...);
                }
            };

        } // namespace operators
    } // namespace filter
} // namespace pmacc
