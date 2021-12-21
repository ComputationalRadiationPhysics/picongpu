/* Copyright 2017-2021 Rene Widera
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

#include <string>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** execute an unary functor if the name is equal
             *
             * @tparam T_Filter filter class (required interface: `getName( )` and default constructor)
             */
            template<typename T_Filter>
            struct ExecuteIfNameIsEqual
            {
                /** evaluate if functor must executed
                 *
                 * @param filterName name of the filter which should started
                 * @param unaryFunctor any unary functor
                 */
                template<typename T_Kernel, typename... T_Args>
                void operator()(std::string filterName, uint32_t const currentStep, T_Kernel const unaryFunctor) const
                {
                    if(filterName == T_Filter::getName())
                        unaryFunctor(particles::filter::IUnary<T_Filter>{currentStep});
                }
            };
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
